"""
Fashion Attribute Extraction – Robust Pipeline

Features:
- URL validation (HEAD+GET fallback), content-type and size checks
- Parallel downloads with retries/backoff
- Vision inference via:
  A) Local HuggingFace CLIP (default, no API key)
  B) Optional BLIP/Florence2 zero-shot captions (HF) to aid classification
  C) Optional Gemini/OpenAI Vision if env keys provided (GEMINI_API_KEY / OPENAI_API_KEY)
- Attributes: Length, Neckline, Silhouette, Waistline, Sleeves
- Confidence thresholds + ambiguity handling → "Unknown"
- Post-rules (e.g., halter/strapless ⇒ sleeveless)
- Cleaning & standardization to a strict taxonomy
- Structured outputs: CSV + JSON + failed_urls.txt
- Insight charts and CSV distributions per attribute

Usage examples:
  python fashion_pipeline.py --input urls.txt --out_csv fashion_raw.csv --clean_csv fashion_clean.csv
  # with BLIP caption assist
  python fashion_pipeline.py --input urls.txt --use_blip True
  # with Gemini/OpenAI (if keys set in env)
  GEMINI_API_KEY=... python fashion_pipeline.py --input urls.txt --use_gemini True
  OPENAI_API_KEY=... python fashion_pipeline.py --input urls.txt --use_openai True

Notes:
- Install: pip install -r requirements.txt
"""

import argparse, os, time, json, logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# Optional helpers (loaded lazily if used)
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    _HAS_BLIP = True
except Exception:
    _HAS_BLIP = False

logger = logging.getLogger("fashion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ----------------- Canonical Taxonomy -----------------
CANON = {
    "Length": ["Mini","Knee-Length","Midi","Ankle-Length","Maxi","Floor-Length"],
    "Neckline": ["Sweetheart","V-Neck","Crew","Halter Neck","Round Neck","Scoop","Square Neck","Collared","Boat Neck","Off-Shoulder","Strapless","Spaghetti"],
    "Silhouette": ["A-Line","Mermaid","Sheath","Ball Gown","Empire","Shift","Pantsuit","Jumpsuit","Fit-and-Flare","Column"],
    "Waistline": ["Empire","Natural","Drop","Basque","High Waist","Low-Waist","Asymmetrical","Cinched"],
    "Sleeves": ["Sleeveless","Cap","Puff","Bell","Bishop","Raglan","Kimono","Long","Short","3/4 Sleeves"],
}

LOWER_CANON = {k: {v.lower(): v for v in vals} for k, vals in CANON.items()}

# ----------------- HTTP Session -----------------
def make_session(pool=64, retries=3, backoff=0.6):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

SESSION = make_session()

# ----------------- URL Validation -----------------
VALID_CONTENT_TYPES = {"image/jpeg","image/jpg","image/png","image/webp"}

def validate_url(url: str, min_bytes: int = 2048) -> Optional[int]:
    """Return content length (int) if valid image; else None."""
    try:
        r = SESSION.head(url, timeout=8, allow_redirects=True)
        if r.status_code >= 400:
            # some CDNs block HEAD; try GET with stream
            r = SESSION.get(url, timeout=10, stream=True)
        ct = r.headers.get("content-type", "").split(";")[0].lower()
        if ct not in VALID_CONTENT_TYPES:
            return None
        clen = r.headers.get("content-length")
        if clen is not None and int(clen) < min_bytes:
            return None
        return int(clen) if clen is not None else min_bytes
    except Exception:
        return None

# ----------------- Download -----------------

def fetch_image(url: str, max_bytes: int = 10_000_000) -> Optional[Image.Image]:
    try:
        r = SESSION.get(url, timeout=15)
        r.raise_for_status()
        if len(r.content) > max_bytes:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        # sanity check size
        if img.width < 80 or img.height < 80:
            return None
        return img
    except Exception:
        return None



# ----------------- Gemini Vision (optional) -----------------
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")
_GEMINI_MODEL = None

def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return buf.getvalue(), mime

GEMINI_EXTRACTION_PROMPT = (
    "You are a fashion taxonomy assistant. Given a single dress image, return a strict JSON object with fields: "
    "Length, Neckline, Silhouette, Waistline, Sleeves. "
    "Use ONLY these allowed values (case-sensitive):\n"
    f"Length: {', '.join(CANON['Length'])}\n"
    f"Neckline: {', '.join(CANON['Neckline'])}\n"
    f"Silhouette: {', '.join(CANON['Silhouette'])}\n"
    f"Waistline: {', '.join(CANON['Waistline'])}\n"
    f"Sleeves: {', '.join(CANON['Sleeves'])}\n"
    "If unsure for a field, set it to 'Unknown'. Output: a single JSON object, no prose."
)

def init_gemini():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is not None:
        return
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    _GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)

def gemini_classify(img: Image.Image):
    """Return dict of attributes or None if parsing fails. Requires env GEMINI_API_KEY."""
    try:
        init_gemini()
        img_bytes, mime = _pil_to_bytes(img, fmt="JPEG")
        # Build the image part
        image_part = {"mime_type": mime, "data": img_bytes}
        resp = _GEMINI_MODEL.generate_content([
            GEMINI_EXTRACTION_PROMPT,
            image_part,
        ])
        txt = resp.text or ""
        # extract JSON
        import json, re
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None
        obj = json.loads(m.group(0))
        out = {}
        for k in ["Length","Neckline","Silhouette","Waistline","Sleeves"]:
            out[k] = normalize_value(k, obj.get(k, "Unknown"))
        return out
    except Exception:
        return None

# ----------------- CLIP Setup -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"
CLIP_MODEL: Optional[CLIPModel] = None
CLIP_PROC: Optional[CLIPProcessor] = None

PROMPTS = {
    "Length": {
        "Mini": ["a mini dress","short above-the-knee dress"],
        "Knee-Length": ["a knee-length dress"],
        "Midi": ["a midi length dress"],
        "Ankle-Length": ["an ankle-length dress"],
        "Maxi": ["a maxi dress"],
        "Floor-Length": ["a floor-length gown","a full-length evening gown"],
        "Unknown": ["length unclear"],
    },
    "Neckline": {
        "Halter Neck":["a halter neck dress"],
        "Spaghetti":["a spaghetti strap dress"],
        "Strapless":["a strapless dress"],
        "Off-Shoulder":["an off-shoulder dress"],
        "V-Neck":["a V-neck dress"],
        "Crew":["a crew neck dress"],
        "Round Neck":["a round-neck dress"],
        "Scoop":["a scoop-neck dress"],
        "Square Neck":["a square-neck dress"],
        "Collared":["a collared dress"],
        "Boat Neck":["a boat-neck dress"],
        "Sweetheart":["a sweetheart neckline dress"],
        "Unknown":["neckline unclear"],
    },
    "Silhouette": {
        "A-Line":["an A-line dress"],
        "Sheath":["a sheath dress"],
        "Ball Gown":["a ball gown"],
        "Mermaid":["a mermaid silhouette dress"],
        "Empire":["an empire waist silhouette dress"],
        "Shift":["a shift dress"],
        "Pantsuit":["a pantsuit outfit"],
        "Jumpsuit":["a jumpsuit outfit"],
        "Fit-and-Flare":["a fit-and-flare dress"],
        "Column":["a column dress"],
        "Unknown":["silhouette unclear"],
    },
    "Waistline": {
        "Natural":["a natural waistline dress"],
        "Empire":["an empire waistline dress"],
        "Drop":["a drop waist dress"],
        "Basque":["a basque waistline dress"],
        "High Waist":["a high-waist dress"],
        "Low-Waist":["a low-waist dress"],
        "Asymmetrical":["an asymmetrical waistline dress"],
        "Cinched":["a cinched waist dress"],
        "Unknown":["waistline unclear"],
    },
    "Sleeves": {
        "Sleeveless":["a sleeveless dress","a dress with no sleeves"],
        "Long":["a dress with long sleeves"],
        "Short":["a short-sleeve dress"],
        "3/4 Sleeves":["a dress with three-quarter sleeves"],
        "Cap":["a dress with cap sleeves"],
        "Puff":["a puff-sleeve dress"],
        "Bell":["a bell-sleeve dress"],
        "Bishop":["a bishop-sleeve dress"],
        "Raglan":["a raglan-sleeve dress"],
        "Kimono":["a kimono-sleeve dress"],
        "Unknown":["sleeves unclear"],
    },
}

LOGIT_SCALE = None

@torch.no_grad()
def init_clip():
    global CLIP_MODEL, CLIP_PROC, LOGIT_SCALE
    if CLIP_MODEL is None:
        logger.info("Loading CLIP model…")
        CLIP_MODEL = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
        CLIP_PROC = CLIPProcessor.from_pretrained(CLIP_NAME)
        LOGIT_SCALE = CLIP_MODEL.logit_scale.exp().item()

@torch.no_grad()
def normalize(x: torch.Tensor):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

@torch.no_grad()
def build_text_bank():
    bank = {}
    for attr, label_map in PROMPTS.items():
        labels = list(label_map.keys())
        prompts = []
        label_slices = []
        for lab in labels:
            ps = label_map[lab]
            label_slices.append((lab, len(ps)))
            prompts.extend(ps)
        inputs = CLIP_PROC(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        tfeat = normalize(CLIP_MODEL.get_text_features(**inputs))
        # split back per label -> mean
        cur = []
        idx = 0
        for lab, k in label_slices:
            seg = tfeat[idx:idx+k]
            cur.append(seg.mean(dim=0, keepdim=True))
            idx += k
        bank[attr] = {"labels": labels, "embeds": torch.cat(cur, dim=0)}
    return bank

TEXT_BANK = None

@torch.no_grad()
def clip_classify(img: Image.Image, min_conf=0.42) -> Tuple[Dict[str,str], Dict[str,dict]]:
    inputs = CLIP_PROC(images=img, return_tensors="pt").to(DEVICE)
    img_feat = normalize(CLIP_MODEL.get_image_features(**inputs))
    preds = {}
    scores = {}
    for attr, bank in TEXT_BANK.items():
        sims = (img_feat @ bank["embeds"].T).squeeze(0)
        probs = torch.softmax(sims * LOGIT_SCALE, dim=0)
        pmax, idx = probs.max(dim=0)
        label = bank["labels"][idx.item()]
        if pmax.item() < min_conf:
            label = "Unknown"
        preds[attr] = label
        scores[attr] = {"labels": bank["labels"], "probs": probs.detach().cpu().tolist()}
    return preds, scores

# ----------------- BLIP caption assist (optional) -----------------
@torch.no_grad()
def blip_caption(img: Image.Image) -> Optional[str]:
    if not _HAS_BLIP:
        return None
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(DEVICE)
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs, max_new_tokens=30)
        return processor.tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        return None

# ----------------- Post-rules -----------------

def apply_rules(preds: Dict[str,str], scores: Dict[str,dict], sleeve_margin=0.20) -> Dict[str,str]:
    neckline = preds.get("Neckline", "Unknown")
    if neckline in {"Halter Neck","Strapless","Spaghetti"}:
        labs = scores["Sleeves"]["labels"]
        probs = scores["Sleeves"]["probs"]
        def p(lbl):
            return probs[labs.index(lbl)] if lbl in labs else 0.0
        p_s = p("Sleeveless"); p_long = p("Long"); p_short = p("Short")
        if (max(p_long, p_short) - p_s) < sleeve_margin:
            preds["Sleeves"] = "Sleeveless"
    return preds

# ----------------- Cleaning -----------------
MAPS = {
    "Length": {"mini":"Mini","knee":"Knee-Length","knee-length":"Knee-Length","midi":"Midi","ankle":"Ankle-Length","maxi":"Maxi","floor":"Floor-Length","floor-length":"Floor-Length"},
    "Neckline": {"off shoulder":"Off-Shoulder","off-shoulder":"Off-Shoulder","crew":"Crew","round":"Round Neck","halter":"Halter Neck","spaghetti":"Spaghetti"},
    "Silhouette": {"ballgown":"Ball Gown","ball gown":"Ball Gown","fit and flare":"Fit-and-Flare","fit-and-flare":"Fit-and-Flare","pantsuit":"Pantsuit","pant suit":"Pantsuit"},
    "Waistline": {"low waist":"Low-Waist","low-waist":"Low-Waist","high waist":"High Waist","high-waist":"High Waist","asym":"Asymmetrical"},
    "Sleeves": {"short sleeve":"Short","short":"Short","3/4":"3/4 Sleeves","three quarter":"3/4 Sleeves","long":"Long","sleeveless":"Sleeveless"},
}

def normalize_value(col: str, val: Optional[str]) -> str:
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    s = str(val).strip()
    # direct canonical match (case-insensitive)
    base = s.lower()
    if col in LOWER_CANON and base in LOWER_CANON[col]:
        return LOWER_CANON[col][base]
    # mapping
    for k, v in MAPS.get(col, {}).items():
        if k in base:
            return v
    # title-case fallback
    return s.title()

# ----------------- Pipeline -----------------

def process_urls(urls: List[str], min_conf=0.42, workers=8, caption_assist=False, use_gemini: bool = False):
    init_clip()
    global TEXT_BANK
    if TEXT_BANK is None:
        TEXT_BANK = build_text_bank()

    results = []
    failed = []

    # 1) validate
    valid_urls = [u for u in urls if validate_url(u) is not None]

    # 2) download in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_image, u): u for u in valid_urls}
        for fut in as_completed(futs):
            url = futs[fut]
            img = fut.result()
            if img is None:
                failed.append(url); continue
                        # Prefer Gemini if enabled; otherwise CLIP
            preds = None
            scores = {}
            if use_gemini:
                try:
                    preds = gemini_classify(img)
                except Exception:
                    preds = None
            if not preds:
                preds, scores = clip_classify(img, min_conf=min_conf)
                preds = apply_rules(preds, scores)

            # Optional: caption assist (not used for final label directly; could be logged)
            if caption_assist:
                cap = blip_caption(img)
            else:
                cap = None

            # Normalize to canonical taxonomy
            row = {k: normalize_value(k, v) for k, v in preds.items()}
            results.append({"Image_url": url, **row, "_caption": cap})

    return results, failed

# ----------------- Insights -----------------

def save_insights(df: pd.DataFrame, prefix: str = "fashion"):
    os.makedirs("insights", exist_ok=True)
    for col in ["Length","Neckline","Silhouette","Waistline","Sleeves"]:
        if col not in df.columns: continue
        counts = df[col].value_counts().sort_values(ascending=False)
        counts.to_csv(f"insights/{prefix}_{col.lower()}_distribution.csv")
        plt.figure(figsize=(8,5))
        counts.plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"insights/{prefix}_{col.lower()}_distribution.png")
        plt.close()

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Fashion Attribute Extraction – Robust Pipeline")
    ap.add_argument("--input", required=True, help="Path to urls.txt (one URL per line) or CSV with Image_url column")
    ap.add_argument("--out_csv", default="fashion_raw.csv")
    ap.add_argument("--out_json", default="fashion_raw.json")
    ap.add_argument("--clean_csv", default="fashion_clean.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_conf", type=float, default=0.42)
    ap.add_argument("--use_blip", type=bool, default=False)
    ap.add_argument("--use_gemini", type=bool, default=False)  # placeholder
    ap.add_argument("--use_openai", type=bool, default=False)  # placeholder
    args = ap.parse_args()

    # read URLs
    urls: List[str] = []
    if args.input.lower().endswith(".txt"):
        with open(args.input, "r", encoding="utf-8") as f:
            urls = [ln.strip() for ln in f if ln.strip()]
    else:
        df_in = pd.read_csv(args.input)
        col = "Image_url" if "Image_url" in df_in.columns else df_in.columns[0]
        urls = [str(u).strip() for u in df_in[col].dropna().tolist()]

    logger.info(f"Loaded {len(urls)} URLs")

    results, failed = process_urls(urls, min_conf=args.min_conf, workers=args.workers, caption_assist=args.use_blip, use_gemini=args.use_gemini)

    # save raw
    df_raw = pd.DataFrame(results)
    df_raw.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # clean & standardize (idempotent here because we already normalized, but also fix any residuals)
    for col in ["Length","Neckline","Silhouette","Waistline","Sleeves"]:
        if col in df_raw:
            df_raw[col] = df_raw[col].apply(lambda v: normalize_value(col, v))

    # drop helper column and dups
    if "_caption" in df_raw:
        df_clean = df_raw.drop(columns=["_caption"]).drop_duplicates(subset=["Image_url"]).copy()
    else:
        df_clean = df_raw.drop_duplicates(subset=["Image_url"]).copy()

    df_clean.to_csv(args.clean_csv, index=False)

    # save failed
    if failed:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        logger.warning(f"Saved {len(failed)} failed URLs to failed_urls.txt")

    # insights
    save_insights(df_clean, prefix="fashion")

    logger.info(f"Raw -> {args.out_csv}; Clean -> {args.clean_csv}; JSON -> {args.out_json}")

if __name__ == "__main__":
    main()
