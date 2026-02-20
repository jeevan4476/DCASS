import json
import nltk
import re  
from pathlib import Path
from nltk.tokenize import sent_tokenize

# --- NLTK Setup (Auto-Download) ---
def ensure_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

ensure_nltk_resources()

class TextChunker:
    def __init__(self, window_size=3, stride=1):
        self.window_size = window_size
        self.stride = stride

    def clean_text(self, text: str) -> str:
        """Removes references, brackets, and extra whitespace."""
        # 1. Remove [References] or [12] style footnotes
        text = re.sub(r'\[.*?\]', '', text)
        
        # 2. Remove "Chapter 1." or "27." numbering at start of lines
        # This handles the "27. If we know..." issue
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # 3. Existing bibliography stripper
        markers = ["references", "bibliography"]
        lower = text.lower()
        for marker in markers:
            idx = lower.rfind(marker)
            if idx != -1 and idx > len(text) * 0.8:
                text = text[:idx]
                
        # 4. Normalize whitespace
        return " ".join(text.split())

    def is_garbage(self, text: str) -> bool:
        """Detects PDF artifacts (e.g. 's p a c e d  t e x t')."""
        tokens = text.split()
        if not tokens: return True
        single_chars = sum(1 for t in tokens if len(t) == 1)
        return (single_chars / len(tokens)) > 0.4

    def process_file(self, file_path: Path) -> list:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            clean_content = self.clean_text(content)
            sentences = sent_tokenize(clean_content)
            
            chunks = []
            for i in range(0, len(sentences) - self.window_size + 1, self.stride):
                window = sentences[i : i + self.window_size]
                chunk_text = " ".join(window)
                
                if not self.is_garbage(chunk_text):
                    chunks.append({
                        "source": file_path.name,
                        "text": chunk_text
                    })
            return chunks
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return []