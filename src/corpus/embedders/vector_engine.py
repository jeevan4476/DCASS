import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

class VectorEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []

    def _load_model(self):
        """Lazy loader for the model to save RAM until needed."""
        if not self.model:
            print(f"Loading Model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, chunks_path: Path, index_output_path: Path):
        """Generates embeddings from a JSON chunks file and saves a FAISS index."""
        self._load_model()
        
        print(f"Loading chunks from {chunks_path}...")
        if not chunks_path.exists():
            print(f"Error: File not found: {chunks_path}")
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if not self.chunks:
            print("Warning: No chunks found to embed.")
            return

        print(f"Embedding {len(self.chunks)} chunks (this may take a while)...")
        texts = [c["text"] for c in self.chunks]
        
        # Generate Embeddings
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )

        # Build FAISS Index (Inner Product for Cosine Similarity)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings.astype("float32"))

        # Save Index
        faiss.write_index(self.index, str(index_output_path))
        print(f"Index saved to {index_output_path}")

    def load_index(self, chunks_path: Path, index_path: Path):
        """Loads an existing index and chunk data."""
        self._load_model()
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file missing: {index_path}")
            
        print("Loading index and chunks...")
        self.index = faiss.read_index(str(index_path))
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query: str, k=3):
        """Finds the semantic match for a query."""
        if not self.index:
            raise Exception("Index not loaded! Call load_index() first.")

        # Embed query
        q_emb = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS
        scores, indices = self.index.search(q_emb.astype("float32"), k)

        results = []
        for rank, idx in enumerate(indices[0]):
            # FAISS returns -1 if no neighbor found
            if idx == -1: continue
            
            results.append({
                "score": float(scores[0][rank]),
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx].get("source", "unknown")
            })
        return results