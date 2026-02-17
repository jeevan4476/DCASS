import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.corpus.embedders.vector_engine import VectorEngine

def main():
    # 1. Setup
    chunks_path = "data/processed/chunks.json"
    index_path = "data/processed/faiss.index"
    
    engine = VectorEngine()
    
    # Check if we need to build first
    if not Path(index_path).exists():
        print("Index not found. Building from chunks...")
        if not Path(chunks_path).exists():
             print("No chunks found! Run chunker.py first.")
             return
        engine.build_index(chunks_path, index_path)
    else:
        engine.load_index(chunks_path, index_path)

    # 2. Define Secret Message
    secret_message = "The troops will attack at dawn"
    
    # 3. "Semantic Chunking" of Secret (Simple split for PoC)
    # In real DCASS, this would be smarter. Here we split by concept manually for demo.
    secret_segments = ["The troops", "will attack", "at dawn"]
    
    print(f"\nSecret Message: '{secret_message}'")
    print(f"Segments: {secret_segments}\n")
    print("-" * 50)
    print("SEARCHING FOR COVERS...")
    print("-" * 50)

    # 4. Find Covers
    cover_letter = []
    for seg in secret_segments:
        # Find best match (k=1)
        match = engine.search(seg, k=1)[0]
        cover_letter.append(match['text'])
        
        print(f"Secret: '{seg}'")
        print(f"Match ({match['score']:.4f}): \"{match['text'][:60]}...\"")
        print("-" * 20)

    # 5. Result
    print("\nGENERATED COVER TEXT:")
    print(" ".join(cover_letter))

if __name__ == "__main__":
    main()