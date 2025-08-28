import argparse, logging, time, json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests, pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import coloredlogs
from tabulate import tabulate
from colorama import Fore, Style
import matplotlib.pyplot as plt

# -------------------- Logging --------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(
    level="INFO", logger=logger,
    fmt="%(asctime)s %(levelname)s %(message)s",
    level_styles={"info":{"color":"green"}, "warning":{"color":"yellow"}, "error":{"color":"red"}}
)

# -------------------- Load CLIP --------------------
logger.info("Loading the CLIP model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_name)

# -------------------- Attribute Labels --------------------
# Canonical labels (what will appear in CSV)
ATTRIBUTE_SETS = {
    "Neckline": ["V-Neck", "Round Neck", "Scoop", "Square Neck", "Collared",
                 "Boat Neck", "Halter Neck", "Sweetheart", "Crew", "Off-shoulder", "Strapless", "Spaghetti"],
    "Silhouette": ["A-Line", "Sheath", "Ball gown", "Mermaid", "Empire", "Shift",
                   "Pantsuit", "Jumpsuit", "Fit-and-flare", "Column"],
    "Waistline": ["Natural", "Empire", "Drop", "Basque", "High Waist",
                  "Low-waist", "Asymmetrical", "Cinched"],
    "Sleeves": ["Sleeveless", "Cap", "Puff", "Bell", "Bishop",
                "Raglan", "Kimono", "Long", "Short", "3/4 Sleeves"],
}

# Multi-template prompts per label to stabilize CLIP
PROMPT_BANK = {
    "Sleeves": {
        "Sleeveless": [
            "a sleeveless dress", "a dress with no sleeves", "a gown without sleeves",
            "a sleeveless top", "a sleeveless outfit"
        ],
        "Long": [
            "a dress with long sleeves", "a long-sleeve dress", "a long-sleeved gown"
        ],
        "Short": [
            "a dress with short sleeves", "a short-sleeve dress", "a short-sleeved top"
        ],
        "3/4 Sleeves": [
            "a dress with three-quarter sleeves", "a 3/4 sleeved dress"
        ],
        "Cap": [
            "a dress with cap sleeves", "a cap-sleeve dress"
        ],
        "Puff": [
            "a dress with puff sleeves", "a puff-sleeve dress"
        ],
        "Bell": [
            "a dress with bell sleeves", "a bell-sleeve dress"
        ],
        "Bishop": [
            "a dress with bishop sleeves", "a bishop-sleeve dress"
        ],
        "Raglan": [
            "a dress with raglan sleeves", "a raglan-sleeve top"
        ],
        "Kimono": [
            "a dress with kimono sleeves", "a kimono-sleeve dress"
        ],
        "Unknown": [
            "sleeves are unclear"
        ]
    },
    "Neckline": {
        "Halter Neck": [
            "a halter neck dress", "a dress with a halter neckline"
        ],
        "Spaghetti": [
            "a dress with spaghetti straps", "a spaghetti strap dress"
        ],
        "Strapless": [
            "a strapless dress", "a dress with no straps"
        ],
        "Off-shoulder": [
            "an off-shoulder dress", "a dress with off-the-shoulder neckline"
        ],
        "V-Neck": ["a V-neck dress","a dress with a V neckline"],
        "Round Neck": ["a round-neck dress","a crew neck dress"],
        "Crew": ["a crew neck top","a crew neck dress"],
        "Scoop": ["a scoop-neck dress"],
        "Square Neck": ["a square-neck dress"],
        "Collared": ["a collared dress","a dress with a shirt collar"],
        "Boat Neck": ["a boat-neck dress"],
        "Sweetheart": ["a sweetheart neckline dress"],
        "Unknown": ["neckline unclear"]
    },
    "Silhouette": {
        "A-Line": ["an A-line dress","an A-line silhouette"],
        "Sheath": ["a sheath dress","a slim sheath silhouette"],
        "Ball gown": ["a ball gown","a ballgown dress"],
        "Mermaid": ["a mermaid dress","a fishtail silhouette"],
        "Empire": ["an empire waist silhouette dress"],
        "Shift": ["a shift dress"],
        "Pantsuit": ["a pantsuit outfit"],
        "Jumpsuit": ["a jumpsuit outfit"],
        "Fit-and-flare": ["a fit-and-flare dress"],
        "Column": ["a column dress","a straight column silhouette"],
        "Unknown": ["silhouette unclear"]
    },
    "Waistline": {
        "Natural": ["a natural waistline dress"],
        "Empire": ["an empire waistline dress"],
        "Drop": ["a drop waist dress"],
        "Basque": ["a basque waistline dress"],
        "High Waist": ["a high-waist dress"],
        "Low-waist": ["a low-waist dress"],
        "Asymmetrical": ["an asymmetrical waistline dress"],
        "Cinched": ["a cinched waist dress"],
        "Unknown": ["waistline unclear"]
    }
}

# -------------------- Helpers --------------------
def _make_session(pool=64, retries=2, backoff=0.5):
    s = requests.Session()
    retry = requests.adapters.Retry(total=retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504])
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

def load_image(url, retries=3, delay=1.0):
    session = _make_session()
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=10); r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            if attempt < retries-1:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(delay)
    logger.error(f"Failed to load image after {retries} attempts: {url}")
    return None

def normalize_tensor(x: torch.Tensor):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

# Precompute text embeddings (average over templates per label)
def build_text_bank():
    text_bank = {}
    with torch.no_grad():
        for attr, labels in ATTRIBUTE_SETS.items():
            label_to_embed = {}
            # use prompt bank if provided, else fall back to generic template
            for label in labels:
                prompts = PROMPT_BANK.get(attr, {}).get(label, [f"A product photo of {label}"])
                inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
                tfeat = model.get_text_features(**inputs)
                tfeat = normalize_tensor(tfeat)
                label_to_embed[label] = tfeat.mean(dim=0, keepdim=True)  # (1, D)
            # stack per-attribute
            label_names = list(label_to_embed.keys())
            embeds = torch.cat([label_to_embed[n] for n in label_names], dim=0)  # (L, D)
            text_bank[attr] = {"labels": label_names, "embeds": embeds}
    return text_bank

TEXT_BANK = build_text_bank()
LOGIT_SCALE = model.logit_scale.exp().item()  # CLIP temperature

def classify_image(image, min_conf=0.42):
    # one image pass
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        img_feat = model.get_image_features(**inputs)
        img_feat = normalize_tensor(img_feat)  # (1, D)

    out = {}
    raw_scores = {}
    for attr, bank in TEXT_BANK.items():
        tfeat = bank["embeds"]  # (L, D)
        sims = (img_feat @ tfeat.T).squeeze(0)  # (L,)
        probs = torch.softmax(sims * LOGIT_SCALE, dim=0)
        pmax, idx = probs.max(dim=0)
        label = bank["labels"][idx.item()]
        if pmax.item() < min_conf:
            label = "Unknown"
        out[attr] = label
        # keep a copy for post rules
        raw_scores[attr] = {"labels": bank["labels"], "probs": probs.detach().cpu().tolist()}
    return out, raw_scores

# Post-rules to fix systematic sleeve mistakes
def postprocess_rules(preds, scores, sleeve_override_margin=0.20):
    neckline = preds.get("Neckline", "Unknown")
    sleeves = preds.get("Sleeves", "Unknown")

    # If halter/strapless/spaghetti → usually sleeveless
    sleeveless_like_necks = {"Halter Neck", "Strapless", "Spaghetti"}
    if neckline in sleeveless_like_necks:
        # only override if not strong evidence for long/short
        sleeves_labels = scores["Sleeves"]["labels"]
        sleeves_probs = scores["Sleeves"]["probs"]
        # get probabilities for critical classes
        def p(lbl):
            return sleeves_probs[sleeves_labels.index(lbl)] if lbl in sleeves_labels else 0.0
        p_sleeveless = p("Sleeveless")
        p_long = p("Long")
        p_short = p("Short")

        # if model not confidently long/short, force Sleeveless
        if (max(p_long, p_short) - p_sleeveless) < sleeve_override_margin:
            preds["Sleeves"] = "Sleeveless"

    return preds

# -------------------- CLEANING --------------------
NECKLINE_MAP = {"halter": "Halter Neck", "halterneck": "Halter Neck", "off shoulder": "Off-shoulder",
                "offshoulder": "Off-shoulder", "round": "Round Neck", "crew": "Crew"}
SILHOUETTE_MAP = {"ballgown": "Ball gown", "ball gown": "Ball gown", "fit flare": "Fit-and-flare",
                  "fit-and-flare": "Fit-and-flare", "pant suit": "Pantsuit", "pantsuit": "Pantsuit"}
WAISTLINE_MAP = {"cinched": "Cinched", "low waist": "Low-waist", "low-waist": "Low-waist",
                 "high waist": "High Waist", "high-waist": "High Waist", "asym": "Asymmetrical"}
SLEEVES_MAP = {"sleeveless": "Sleeveless", "short sleeve": "Short", "short": "Short",
               "3/4": "3/4 Sleeves", "three quarter": "3/4 Sleeves", "long": "Long"}

def normalize_val(value, mapping_dict):
    if pd.isna(value): return "NA"
    val = str(value).strip().lower()
    for k, standard in mapping_dict.items():
        if k in val: return standard
    return value.title() if value else "NA"

def clean_dataframe(df):
    for col, m in [("Neckline", NECKLINE_MAP), ("Silhouette", SILHOUETTE_MAP),
                   ("Waistline", WAISTLINE_MAP), ("Sleeves", SLEEVES_MAP)]:
        if col in df:
            df[col] = df[col].apply(lambda x: normalize_val(x, m))
    df = df.drop_duplicates(subset=["Image_url"])
    if {"Image_url"}.issubset(df.columns):
        df = df.dropna(subset=["Image_url"])
    return df

# -------------------- Insights --------------------
def generate_insights(df, output_prefix="fashion"):
    insights = {}
    for col in ["Neckline", "Silhouette", "Waistline", "Sleeves"]:
        if col not in df: continue
        counts = df[col].value_counts().sort_values(ascending=False)
        insights[col] = counts.to_dict()
        counts.to_csv(f"{output_prefix}_{col.lower()}_distribution.csv")
        plt.figure(figsize=(8,5))
        counts.plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{col.lower()}_distribution.png")
        plt.close()
    logger.info("Summary of attribute distributions:")
    for k, v in insights.items():
        logger.info(f"{k}: {list(v.items())[:5]} ...")
    return insights

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Fashion Attribute Extraction + Cleaning + Insights (CLIP improved)")
    ap.add_argument("--input", required=True, help="Path to urls.txt (one URL per line)")
    ap.add_argument("--output_csv", default="fashion_attributes.csv")
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--cleaned_csv", default="fashion_attributes_cleaned.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_conf", type=float, default=0.25, help="Min confidence to accept a label; else Unknown")
    ap.add_argument("--sleeve_override_margin", type=float, default=0.20,
                   help="If neckline suggests sleeveless, require this prob margin to keep long/short")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip()]
    logger.info(f"Loaded {len(urls)} image URLs")

    # Resume if CSV exists
    try:
        existing = pd.read_csv(args.output_csv)
        processed_urls = set(existing["Image_url"].tolist())
        results = existing.to_dict(orient="records")
        logger.info(f"Resuming: {len(processed_urls)} already processed")
    except Exception:
        processed_urls, results = set(), []
        logger.info("No previous checkpoint; starting fresh")

    failed_urls = []
    stats = {"processed": 0, "skipped": len(processed_urls), "failed": 0}
    to_process = [u for u in urls if u not in processed_urls]

    for i in tqdm(range(0, len(to_process), args.batch_size), desc="Processing Batches"):
        batch_urls = to_process[i:i+args.batch_size]

        # download in parallel
        images = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(load_image, u): u for u in batch_urls}
            for fut in as_completed(futs):
                url = futs[fut]
                img = fut.result()
                if img: images[url] = img
                else:
                    failed_urls.append(url)
                    stats["failed"] += 1

        # classify
        for url, img in images.items():
            preds, scores = classify_image(img, min_conf=args.min_conf)
            preds = postprocess_rules(preds, scores, sleeve_override_margin=args.sleeve_override_margin)
            results.append({"Image_url": url, **preds})
            stats["processed"] += 1

        # checkpoint
        df_ckpt = pd.DataFrame(results); df_ckpt.to_csv(args.output_csv, index=False)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
        time.sleep(0.02)

    if failed_urls:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed_urls))
        logger.warning(f"Saved {len(failed_urls)} failed URLs")

    # Clean and save
    raw_df = pd.DataFrame(results)
    cleaned_df = clean_dataframe(raw_df)
    cleaned_df.to_csv(args.cleaned_csv, index=False)
    logger.info(f"✅ Cleaned data saved to {args.cleaned_csv}")

    # Insights
    _ = generate_insights(cleaned_df, output_prefix="fashion")

    # Summary
    summary = [
        [Fore.GREEN + "✅ Processed" + Style.RESET_ALL, stats["processed"]],
        [Fore.YELLOW + "⚠️ Skipped" + Style.RESET_ALL, stats["skipped"]],
        [Fore.RED + "❌ Failed" + Style.RESET_ALL, stats["failed"]],
        [Fore.CYAN + "📊 Total" + Style.RESET_ALL, stats["processed"] + stats["skipped"] + stats["failed"]],
    ]
    print("\n" + tabulate(summary, headers=["Status", "Count"], tablefmt="fancy_grid"))
    logger.info(f"Saved raw results to {args.output_csv}, cleaned results to {args.cleaned_csv}")

if __name__ == "__main__":
    main()
