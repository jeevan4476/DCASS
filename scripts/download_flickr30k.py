from datasets import load_dataset
import os

dataset = load_dataset("nlphuji/flickr30k", split="test")

output_dir = "data/raw/flickr30k"
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

captions_path = os.path.join(output_dir, "captions.txt")

with open(captions_path, "w", encoding="utf-8") as cap_file:
    for i, item in enumerate(dataset):
        image = item["image"]
        caption = item["caption"]

        image.save(os.path.join(images_dir, f"img_{i:05d}.jpg"))
        cap_file.write(caption + "\n")

print("Dataset saved successfully.")
