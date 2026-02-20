import sys
import os
from pathlib import Path

# Add project root to python path to import src modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.corpus.preprocessors.chunker import TextChunker
from src.corpus.embedders.vector_engine import VectorEngine

def main():
    # Define relative paths
    raw_text_dir = project_root / "data/raw/text/wikipedia  "
    chunks_file = project_root / "data/processed/chunks.json"
    index_file = project_root / "data/processed/faiss.index"

    # --- Step 1: Preprocessing ---
    print("\n--- STEP 1: PREPROCESSING ---")
    chunker = TextChunker()
    all_chunks = []
    
    # Process all text files
    files = list(raw_text_dir.glob("*.txt"))
    if not files:
        print(f"No text files found in {raw_text_dir}!")
        print("Please add some .txt books there first.")
        return

    for txt_file in files:
        print(f"Processing {txt_file.name}...")
        all_chunks.extend(chunker.process_file(txt_file))

    # Save chunks
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Saved {len(all_chunks)} chunks.")

    # --- Step 2: Build Index ---
    print("\n--- STEP 2: BUILDING INDEX ---")
    engine = VectorEngine()
    engine.build_index(chunks_file, index_file)

    # --- Step 3: Test Search ---
    print("\n--- STEP 3: TESTING SEARCH ---")
    test_query = "The army attacked at dawn"
    print(f"Query: '{test_query}'")
    results = engine.search(test_query)
    
    for r in results:
        print(f"Match ({r['score']:.4f}): {r['text'][:100]}...")

if __name__ == "__main__":
    main()