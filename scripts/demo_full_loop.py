import sys
from pathlib import Path

# Add project root to python path so we can import 'src'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.corpus.embedders.vector_engine import VectorEngine
from src.engine.decoder import SemanticDecoder

def main():
    print("DCASS FULL LOOP DEMO (Encoder -> Decoder)")
    print("------------------------------------------------")

    # 1. Setup Paths
    chunks_path = project_root / "data/processed/chunks.json"
    index_path = project_root / "data/processed/faiss.index"
    
    # Check if data exists
    if not index_path.exists():
        print("Index not found! Run 'python scripts/run_pipeline.py' first.")
        return

    print("\nInitializing Alice (Encoder)...")
    alice_engine = VectorEngine()
    alice_engine.load_index(chunks_path, index_path)

    secret_message = "The army attacked at dawn"
    print(f"Secret Message: '{secret_message}'")
    
    # Alice searches for the best cover text
    print("   -> Alice is searching for a cover text...")
    results = alice_engine.search(secret_message, k=1)
    best_cover = results[0]
    
    print(f"Alice Sends:    '{best_cover['text'][:100]}...'")
    print(f"   (Source: {best_cover['source']} | Similarity: {best_cover['score']:.4f})")


    # Alice posts the cover text to a public forum. Bob sees it.
    received_text = best_cover['text']


    print("\nInitializing Bob (Decoder)...")
    bob_decoder = SemanticDecoder(index_path, chunks_path)
    
    print("   -> Bob received the text. Decoding...")
    decoded_payload = bob_decoder.decode(received_text)
    
    # Bob compares the received meaning against the secret (for the demo only)
    # In reality, Bob wouldn't know the secret, but this proves the system works.
    from sentence_transformers import util
    
    # 1. Embed Original Secret
    secret_vec = alice_engine.model.encode(secret_message, convert_to_tensor=True)
    # 2. Embed Received Cover
    cover_vec = bob_decoder.engine.model.encode(received_text, convert_to_tensor=True)
    
    # 3. Calculate Similarity
    sim_score = util.cos_sim(secret_vec, cover_vec).item()
    
    print(f"\n[Decoded Payload]: {decoded_payload}")
    print("-" * 40)
    print(f"SUCCESS METRIC:")
    print(f"   Original Secret: '{secret_message}'")
    print(f"   Recovered Meaning: {sim_score*100:.2f}% Match")
    
    if sim_score > 0.5:
        print("   Result: HIGH FIDELITY TRANSMISSION")
    else:
        print("   Result: LOW FIDELITY (Needs larger corpus)")

if __name__ == "__main__":
    main()