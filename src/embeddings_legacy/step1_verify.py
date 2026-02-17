import csv
from pathlib import Path

CAPTIONS_CSV = Path("data/metadata/captions.csv")
IMAGES_DIR = Path("data/images")
OUTPUT_CSV = Path("data/metadata/verified_captions.csv")

valid_rows = []
removed_rows = 0

with CAPTIONS_CSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row["image_id"]
        image_path = IMAGES_DIR / f"{image_id}.jpg"

        if image_path.exists():
            valid_rows.append(row)
        else:
            removed_rows += 1

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "caption"])
    writer.writeheader()
    writer.writerows(valid_rows)

print(f"Original rows : {len(valid_rows) + removed_rows}")
print(f"Kept rows     : {len(valid_rows)}")
print(f"Removed rows  : {removed_rows}")
print(f"Saved to      : {OUTPUT_CSV}")
