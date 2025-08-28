import argparse
import logging
import sys
import time
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import coloredlogs
from tabulate import tabulate
from colorama import Fore, Style
import torch
import matplotlib.pyplot as plt

# -------------------- Logging with tqdm support --------------------
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())

coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s %(levelname)s %(message)s",
    level_styles={
        "debug": {"color": "blue"},
        "info": {"color": "green"},
        "warning": {"color": "yellow"},
        "error": {"color": "red"},
        "critical": {"color": "magenta", "bold": True},
    },
    field_styles={
        "asctime": {"color": "cyan"},
        "levelname": {"bold": True},
    }
)

# -------------------- Load CLIP --------------------
logger.info("Loading the CLIP model")
torch.set_num_threads(torch.get_num_threads())  # use all CPU cores
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -------------------- Attribute Labels --------------------
ATTRIBUTE_SETS = {
    "Neckline": ["V-Neck", "Round Neck", "Scoop", "Square Neck", "Collared",
                 "Boat Neck", "Halter Neck", "Sweetheart", "Crew", "Off-shoulder", "Unknown"],
    "Silhouette": ["A-Line", "Sheath", "Ball gown", "Mermaid", "Empire", "Shift",
                   "Pantsuit", "Jumpsuit", "Fit-and-flare", "Column", "Unknown"],
    "Waistline": ["Natural", "Empire", "Drop", "Basque", "High Waist",
                  "Low-waist", "Asymmetrical", "Cinched", "Unknown"],
    "Sleeves": ["Sleeveless", "Cap", "Puff", "Bell", "Bishop",
                "Raglan", "Kimono", "Long", "Short", "3/4 Sleeves", "Unknown"],
}

# -------------------- Helpers --------------------
def classify_all(image):
    attributes = {}
    for attr_name, labels in ATTRIBUTE_SETS.items():
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        attributes[attr_name] = labels[probs.argmax().item()]
    return attributes


def load_image(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(delay)
    logger.error(f"Failed to load image after {retries} attempts: {url}")
    return None


def save_checkpoint(results, output_csv, output_json=None):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

# -------------------- CLEANING --------------------
NECKLINE_MAP = {"halter": "Halter Neck", "halterneck": "Halter Neck", "off shoulder": "Off-shoulder",
                "offshoulder": "Off-shoulder", "round": "Round Neck", "crew": "Round Neck"}
SILHOUETTE_MAP = {"ballgown": "Ball gown", "ball gown": "Ball gown", "fit flare": "Fit-and-flare",
                  "fit-and-flare": "Fit-and-flare", "pant suit": "Pantsuit", "pantsuit": "Pantsuit"}
WAISTLINE_MAP = {"cinched": "Cinched", "low waist": "Low-waist", "low-waist": "Low-waist",
                 "high waist": "High Waist", "high-waist": "High Waist", "asym": "Asymmetrical"}
SLEEVES_MAP = {"sleeveless": "Sleeveless", "short sleeve": "Short", "short": "Short",
               "3/4": "3/4 Sleeves", "three quarter": "3/4 Sleeves", "long": "Long"}


def normalize(value, mapping_dict):
    if pd.isna(value):
        return "NA"
    val = str(value).strip().lower()
    for key, standard in mapping_dict.items():
        if key in val:
            return standard
    return value.title() if value else "NA"


def clean_dataframe(df):
    df["Neckline"] = df["Neckline"].apply(lambda x: normalize(x, NECKLINE_MAP))
    df["Silhouette"] = df["Silhouette"].apply(lambda x: normalize(x, SILHOUETTE_MAP))
    df["Waistline"] = df["Waistline"].apply(lambda x: normalize(x, WAISTLINE_MAP))
    df["Sleeves"] = df["Sleeves"].apply(lambda x: normalize(x, SLEEVES_MAP))
    df = df.drop_duplicates(subset=["Image_url"])
    df = df.dropna(subset=["Image_url", "Neckline", "Silhouette"])
    return df

# -------------------- Insights & Visualization --------------------
def generate_insights(df, output_prefix="fashion"):
    insights = {}
    for col in ["Neckline", "Silhouette", "Waistline", "Sleeves"]:
        counts = df[col].value_counts().sort_values(ascending=False)
        insights[col] = counts.to_dict()
        
        # Save summary CSV per attribute
        counts.to_csv(f"{output_prefix}_{col.lower()}_distribution.csv")

        # Save bar chart
        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{col.lower()}_distribution.png")
        plt.close()

    # Print summary
    logger.info("Summary of attribute distributions:")
    for k, v in insights.items():
        logger.info(f"{k}: {list(v.items())[:5]} ...")

    return insights

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Fashion Attribute Extraction + Cleaning + Insights")
    parser.add_argument("--input", required=True, help="Path to urls.txt (one URL per line)")
    parser.add_argument("--output_csv", default="fashion_attributes.csv", help="Raw output CSV path")
    parser.add_argument("--output_json", default=None, help="Optional JSON output path")
    parser.add_argument("--cleaned_csv", default="fashion_attributes_cleaned.csv", help="Final cleaned CSV path")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images per batch")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(urls)} image URLs")

    results = []
    failed_urls = []

    try:
        existing = pd.read_csv(args.output_csv)
        processed_urls = set(existing["Image_url"].tolist())
        results = existing.to_dict(orient="records")
        logger.info(f"Resuming: {len(processed_urls)} already processed")
    except Exception:
        processed_urls = set()
        logger.info("No previous checkpoint found, starting fresh")

    stats = {"processed": 0, "skipped": len(processed_urls), "failed": 0}
    urls_to_process = [u for u in urls if u not in processed_urls]

    for i in tqdm(range(0, len(urls_to_process), args.batch_size), desc="Processing Batches"):
        batch_urls = urls_to_process[i:i + args.batch_size]

        images = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_url = {executor.submit(load_image, url): url for url in batch_urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                img = future.result()
                if img:
                    images[url] = img
                else:
                    failed_urls.append(url)
                    stats["failed"] += 1

        for url, img in images.items():
            attrs = classify_all(img)
            results.append({"Image_url": url, **attrs})
            stats["processed"] += 1

        save_checkpoint(results, args.output_csv, args.output_json)
        time.sleep(0.05)

    if failed_urls:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed_urls))
        logger.warning(f"Saved {len(failed_urls)} failed URLs")

    logger.info("Cleaning and standardizing extracted data...")
    raw_df = pd.DataFrame(results)
    cleaned_df = clean_dataframe(raw_df)
    cleaned_df.to_csv(args.cleaned_csv, index=False)
    logger.info(f"✅ Cleaned data saved to {args.cleaned_csv}")

    # Generate insights and visualizations
    insights = generate_insights(cleaned_df, output_prefix="fashion")

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
