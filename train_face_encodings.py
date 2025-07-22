import os
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO
import random

# Define random human names and object categories
human_names = ["Elon Musk", "Sundar Pichai", "Emma Watson", "Tom Cruise", "Narendra Modi"]
object_categories = ["laptop", "bicycle", "coffee mug", "smartphone", "chair"]

# Combine all into one list for random sampling
search_terms = random.sample(human_names + object_categories, k=5)

# Function to download images using ddgs
def download_images(search_term, limit=1000, save_dir="training_images"):
    folder = os.path.join(save_dir, search_term.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    saved_files = []

    with DDGS() as ddgs:
        results = ddgs.images(search_term, max_results=limit)
        for idx, result in enumerate(results):
            try:
                image_url = result["image"]  # ✅ use dictionary key, not attribute
                response = requests.get(image_url, timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                file_path = os.path.join(folder, f"{search_term.replace(' ', '_')}_{idx}.jpg")
                image.save(file_path)
                saved_files.append(file_path)
                print(f"✅ Saved: {file_path}")
            except Exception as e:
                print(f"❌ Failed to download image {idx} for {search_term}: {e}")
    return saved_files


# Download images for each term
download_summary = {}
for term in search_terms:
    print(f"🔍 Downloading images for: {term}")
    download_summary[term] = download_images(term, limit=1000)

print("🎉 Download summary:", download_summary.keys())
