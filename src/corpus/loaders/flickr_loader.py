"""
Flickr8k/30k Dataset Loader

Loads images and their captions from the Flickr8k or Flickr30k dataset.
Supports multiple directory structures and caption file formats.
"""

from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
from collections import defaultdict

from .base_loader import BaseLoader


class FlickrLoader(BaseLoader):
    """
    Loader for Flickr8k/Flickr30k datasets.
    
    Supports multiple directory structures:
    
    Structure 1 (standard):
        flickr8k/
        ├── images/
        │   ├── 1000268201_693b08cb0e.jpg
        │   └── ...
        └── captions.txt
    
    Structure 2 (Flickr8k.token.txt format):
        flickr8k/
        ├── images/
        │   └── ...
        └── Flickr8k.token.txt
    
    The Flickr8k.token.txt format:
        1000268201_693b08cb0e.jpg#0	A child in a pink dress...
        1000268201_693b08cb0e.jpg#1	A girl going into a wooden...
    
    Example:
        >>> loader = FlickrLoader("data/raw/flickr8k")
        >>> for item in loader.load():
        ...     print(item["id"], item["captions"])
    """
    
    def __init__(
        self,
        source_path: Path,
        images_dir: Optional[Path] = None,
        captions_file: Optional[Path] = None
    ):
        """
        Initialize the Flickr loader.
        
        Args:
            source_path: Path to flickr8k root directory
            images_dir: Optional custom path to images directory
            captions_file: Optional custom path to captions file
        """
        source_path = Path(source_path)
        super().__init__(source_path)
        
        # Find images directory
        if images_dir is not None:
            self.images_dir = Path(images_dir)
        else:
            # Try common names
            for name in ["images", "Images", "Flicker8k_Dataset", "Flickr8k_Dataset"]:
                candidate = source_path / name
                if candidate.exists():
                    self.images_dir = candidate
                    break
            else:
                self.images_dir = source_path / "images"
        
        # Find captions file
        if captions_file is not None:
            self.captions_file = Path(captions_file)
        else:
            # Try common names
            for name in ["Flickr8k.token.txt", "captions.txt", "text/Flickr8k.token.txt"]:
                candidate = source_path / name
                if candidate.exists():
                    self.captions_file = candidate
                    break
            else:
                self.captions_file = source_path / "Flickr8k.token.txt"
        
        # Load captions into memory
        self._captions: Dict[str, List[str]] = {}
        self._image_ids: List[str] = []
        self._loaded = False
    
    def _ensure_loaded(self) -> None:
        """Ensure captions are loaded."""
        if self._loaded:
            return
        self._load_captions()
        self._loaded = True
    
    def _load_captions(self) -> None:
        """Load captions from file and group by image ID."""
        if not self.captions_file.exists():
            print(f"Warning: Captions file not found: {self.captions_file}")
            print("Will generate placeholder captions from filenames.")
            self._load_from_images_only()
            return
        
        captions_by_image: Dict[str, List[str]] = defaultdict(list)
        
        # Detect file format
        with open(self.captions_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        
        if "#" in first_line and "\t" in first_line:
            # Flickr8k.token.txt format: image.jpg#0\tcaption
            self._load_token_format(captions_by_image)
        elif "," in first_line:
            # CSV format: image,caption
            self._load_csv_format(captions_by_image)
        else:
            print(f"Warning: Unknown caption file format in {self.captions_file}")
            self._load_from_images_only()
            return
        
        self._captions = dict(captions_by_image)
        self._image_ids = sorted(self._captions.keys())
        
        print(f"Loaded {len(self._image_ids)} images with {sum(len(c) for c in self._captions.values())} captions")
    
    def _load_token_format(self, captions_by_image: Dict[str, List[str]]) -> None:
        """Load Flickr8k.token.txt format: image.jpg#0\tcaption"""
        with open(self.captions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: image_name#idx\tcaption
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                
                image_part = parts[0].strip()
                caption = parts[1].strip()
                
                # Extract image filename (remove #idx suffix)
                if "#" in image_part:
                    image_file = image_part.split("#")[0]
                else:
                    image_file = image_part
                
                # Use filename as ID
                image_id = Path(image_file).stem
                captions_by_image[image_id].append(caption)
    
    def _load_csv_format(self, captions_by_image: Dict[str, List[str]]) -> None:
        """Load CSV format: image,caption"""
        with open(self.captions_file, "r", encoding="utf-8") as f:
            # Skip header
            header = f.readline()
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                
                image_file = parts[0].strip()
                caption = parts[1].strip()
                
                image_id = Path(image_file).stem
                captions_by_image[image_id].append(caption)
    
    def _load_from_images_only(self) -> None:
        """Load image IDs from directory when no captions file available."""
        if not self.images_dir.exists():
            print(f"Warning: Images directory not found: {self.images_dir}")
            return
        
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in self.images_dir.glob(ext):
                image_id = img_path.stem
                # Generate placeholder caption from filename
                caption = image_id.replace("_", " ")
                self._captions[image_id] = [caption]
        
        self._image_ids = sorted(self._captions.keys())
        print(f"Loaded {len(self._image_ids)} images from directory (no captions file)")
    
    def load(self) -> Iterator[Dict[str, Any]]:
        """
        Yield image items with their captions.
        
        Yields:
            Dict with keys:
                - id: Image ID (filename without extension)
                - image_path: Path to the image file (string)
                - content: Same as image_path for compatibility
                - caption: First caption
                - captions: All captions for this image
        """
        self._ensure_loaded()
        
        for image_id in self._image_ids:
            # Find the image file
            image_path = self._find_image(image_id)
            
            if image_path is None:
                continue
            
            captions = self._captions.get(image_id, [])
            
            yield {
                "id": image_id,
                "image_path": str(image_path),
                "content": str(image_path),
                "caption": captions[0] if captions else "",
                "captions": captions,
            }
    
    def _find_image(self, image_id: str) -> Optional[Path]:
        """Find image file by ID, trying common extensions."""
        extensions = [".jpg", ".jpeg", ".png", ".webp"]
        
        for ext in extensions:
            path = self.images_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        
        return None
    
    def get_captions(self, image_id: str) -> List[str]:
        """Get captions for a specific image."""
        self._ensure_loaded()
        return self._captions.get(image_id, [])
    
    def __len__(self) -> int:
        """Return the number of images."""
        self._ensure_loaded()
        return len(self._image_ids)
    
    @property
    def modality(self) -> str:
        """Return the modality type."""
        return "image"
    
    @property
    def image_ids(self) -> List[str]:
        """Return list of all image IDs."""
        self._ensure_loaded()
        return self._image_ids.copy()
