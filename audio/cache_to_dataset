from datasets import load_dataset

CACHE_DIR = r"D:\DCASS\3gone\raw_audio"
SAVE_DIR  = r"D:\DCASS\3gone\hf_audio_dataset"

dataset = load_dataset(
    "GrigoriiA/libretta-tts-merged-dataset-audio-L10k",
    split="train",
    cache_dir=CACHE_DIR
)

dataset.save_to_disk(SAVE_DIR)

print("✅ Dataset converted & saved at:", SAVE_DIR)
