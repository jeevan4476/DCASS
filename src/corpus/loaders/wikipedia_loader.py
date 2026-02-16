"""
Wikipedia Text Loader

Loads and processes Wikipedia text dumps for use as a text corpus.
"""

from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import re

from .base_loader import BaseLoader


class WikipediaLoader(BaseLoader):
    """
    Loader for Wikipedia text corpus.
    
    Expected directory structure:
        wikipedia/
        ├── article1.txt
        ├── article2.txt
        └── ...
    
    Each .txt file contains the text of a Wikipedia article.
    
    Attributes:
        source_path: Path to the directory containing text files
        extensions: List of file extensions to load
        min_length: Minimum article length in characters
        
    Example:
        >>> loader = WikipediaLoader("data/raw/wikipedia")
        >>> for item in loader.load():
        ...     print(item["id"], len(item["content"]))
    """
    
    def __init__(
        self,
        source_path: Path,
        extensions: Optional[List[str]] = None,
        min_length: int = 100
    ):
        """
        Initialize the Wikipedia loader.
        
        Args:
            source_path: Path to directory containing text files
            extensions: List of file extensions to load (default: [".txt", ".md"])
            min_length: Minimum article length in characters to include
        """
        super().__init__(source_path)
        
        self.extensions = extensions or [".txt", ".md"]
        self.min_length = min_length
        
        # Discover all text files
        self._files: List[Path] = self._discover_files()
    
    def _discover_files(self) -> List[Path]:
        """Find all text files in the source directory."""
        files = []
        
        for ext in self.extensions:
            files.extend(self.source_path.glob(f"*{ext}"))
            files.extend(self.source_path.glob(f"**/*{ext}"))
        
        # Remove duplicates and sort
        files = sorted(set(files))
        
        print(f"Discovered {len(files)} text files")
        return files
    
    def load(self) -> Iterator[Dict[str, Any]]:
        """
        Yield text articles from the corpus.
        
        Yields:
            Dict with keys:
                - id: Article ID (derived from filename)
                - content: Clean text content
                - metadata: Dict containing source file info
        """
        for file_path in self._files:
            try:
                content = self._load_and_clean(file_path)
                
                # Skip short articles
                if len(content) < self.min_length:
                    continue
                
                yield {
                    "id": file_path.stem,
                    "content": content,
                    "metadata": {
                        "filename": file_path.name,
                        "path": str(file_path),
                        "length": len(content),
                    }
                }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    def _load_and_clean(self, file_path: Path) -> str:
        """Load and clean text from a file."""
        # Read content
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Clean the text
        content = self._clean_text(content)
        
        return content
    
    def _clean_text(self, text: str) -> str:
        """
        Clean Wikipedia text by removing common artifacts.
        
        Removes:
            - References like [1], [2], etc.
            - Wikipedia markup artifacts
            - Excessive whitespace
            - Bibliography/References sections
        """
        # Remove reference numbers [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove bracket references [citation needed], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove Wikipedia-style headers (== Section ==)
        text = re.sub(r'={2,}.*?={2,}', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Truncate at common end sections
        end_markers = [
            "references", "bibliography", "external links",
            "see also", "further reading", "notes"
        ]
        lower_text = text.lower()
        for marker in end_markers:
            # Look for marker in the last 20% of text
            cutoff = int(len(text) * 0.8)
            idx = lower_text.rfind(marker)
            if idx > cutoff:
                text = text[:idx]
                break
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def __len__(self) -> int:
        """Return the number of text files."""
        return len(self._files)
    
    @property
    def modality(self) -> str:
        """Return the modality type."""
        return "text"
