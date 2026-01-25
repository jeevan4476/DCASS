from pathlib import Path
from src.corpus.embedders.vector_engine import VectorEngine

class SemanticDecoder:
    def __init__(self, index_path: Path, chunks_path: Path, model_name="all-MiniLM-L6-v2"):
        # Bob needs the same 'language' (Model + Index) as Alice to understand the message
        self.engine = VectorEngine(model_name)
        self.engine.load_index(chunks_path, index_path)

    def decode(self, cover_text: str) -> str:
        """
        Receives 'Cover Text' and extracts the semantic meaning.
        """
        # 1. (Optional) Verify this text actually exists in our valid corpus
        # matches = self.engine.search(cover_text, k=1)
        # if matches[0]['score'] < 0.99:
        #    return "[WARNING: Tampered/Unknown Text]"
        
        # 2. In this Semantic Steganography PoC, the 'meaning' is the vector itself.
        # We return the text tagged as 'Decoded' to show Bob processed it.
        return cover_text