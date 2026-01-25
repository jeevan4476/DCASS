import os
import time
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_wikipedia_subset(target_size_gb=5):
    """
    Downloads clean Wikipedia articles until target size (GB) is reached.
    Optimized to minimize CPU usage during download.
    """
    output_dir = Path("data/raw/text/wikipedia")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Wikipedia Corpus (Target: {target_size_gb} GB)...")
    
    try:
        # Load dataset in streaming mode
        dataset = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en", 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    target_bytes = target_size_gb * 1024 * 1024 * 1024
    
    chunk_id = 0
    current_chunk_text = []
    
    # Trackers
    total_downloaded_bytes = 0
    current_chunk_bytes = 0  # <--- Running counter (Faster!)
    chunk_limit = 100 * 1024 * 1024  # 100MB
    
    # Progress Bar
    pbar = tqdm(total=target_size_gb, unit="GB", desc="Downloading", miniters=1)
    start_time = time.time()
    
    for article in dataset:
        text = article['text']
        
        # Calculate size once
        text_bytes = len(text.encode('utf-8'))
        
        # Update trackers
        current_chunk_text.append(text)
        current_chunk_bytes += text_bytes
        total_downloaded_bytes += text_bytes
        
        # Update Progress Bar
        pbar.update(text_bytes / (1024**3))
        
        # Check limit using the fast counter
        if current_chunk_bytes >= chunk_limit:
            save_chunk(output_dir, chunk_id, current_chunk_text)
            
            # Reset chunk trackers
            chunk_id += 1
            current_chunk_text = []
            current_chunk_bytes = 0
            
        # Stop condition
        if total_downloaded_bytes >= target_bytes:
            print(f"\n\nReached target size: {target_size_gb} GB")
            break
    
    # Save remainder
    if current_chunk_text:
        save_chunk(output_dir, chunk_id, current_chunk_text)

    pbar.close()
    duration = (time.time() - start_time) / 60
    print(f"Done! Downloaded {chunk_id + 1} chunks in {duration:.1f} minutes.")
    print(f"Files ready at: {output_dir.absolute()}")

def save_chunk(directory, chunk_id, texts):
    filename = directory / f"wiki_chunk_{chunk_id:04d}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))
    except Exception as e:
        print(f"Error saving chunk {chunk_id}: {e}")

if __name__ == "__main__":
    # If your internet is slow, you can test with 0.5 first
    download_wikipedia_subset(target_size_gb=0.5)