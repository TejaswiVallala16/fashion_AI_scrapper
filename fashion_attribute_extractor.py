from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import pandas as pd
import json
from io import BytesIO
import argparse
import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Load CLIP model
logging.info("Loading the CLIP model")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_attribute(image, labels):
    """Classify the best matching label for the given image."""
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return labels[probs.argmax().item()]

def validate_image(url):
    logging.info(f"Validating {url}")
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            img.verify()
            return True
    except:
        return False
    return False

def extract_fashion_attributes(image_url):
    # Load image
    logging.info(f"Extracting attributes for {image_url}")
    if validate_image(image_url):
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Candidate label sets
        neckline_labels = ["V-Neck", "Round Neck","Scoop", "Square Neck", "Collared", "Boat Neck", "Halter Neck","Sweetheart","Crew","Off-shoulder", "Unknown"]
        silhouette_labels = ["A-Line", "Sheath", "Ball gown", "Mermaid","Empire", "Shift","Pantsuit", "Jumpsuit","Fit-and-flare","Column", "Unknown"]
        waistline_labels = ["Natural", "Empire", "Drop","Basque", "High Waist","Low-waist", "Asymmetrical","Cinched", "Unknown"]
        sleeve_labels = ["Sleeveless", "Cap", "Puff","Bell","Bishop","Raglan","Kimono","Long","Short", "Unknown", "3/4 Sleeves"]

        # Run classification
        attributes = {
            "neckline": classify_attribute(image, neckline_labels),
            "silhouette": classify_attribute(image, silhouette_labels),
            "waistline": classify_attribute(image, waistline_labels),
            "sleeves": classify_attribute(image, sleeve_labels),
        }

    else:
        attributes = {
            "neckline": None,
            "silhouette": None,
            "waistline": None,
            "sleeves": None,
        }
    return attributes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion Attribute Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Path to urls.txt or a CSV containing an 'image_url' column")
    parser.add_argument("--output_csv", default="fashion_attributes.csv", help="Output CSV path")
    parser.add_argument("--output_json", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        header = f.read(4096)
        f.seek(0)
        urls = [line.strip() for line in f if line.strip()]
        logging.info(f"Processing {len(urls)} Image Urls")
    results = []
    for url in urls:
        result = extract_fashion_attributes(url)
        results.append({"Image_url": url, "neckline":result['neckline'],"silhouette": result['silhouette'], "waistline":result['waistline'],"sleeves": result['sleeves']})
        time.sleep(0.2)

        logging.info(f"Retrived data for {len(results)} Image Urls")
    try:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
    except:
        logging.error("Got error while saving the data")
    logging.info(f"Saved results to {args.output_csv} and {args.output_json}")
