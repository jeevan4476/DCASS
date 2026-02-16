"""
Image Embedder using OpenAI CLIP

Generates dense vector embeddings for images (and text) using CLIP.
This enables cross-modal search: find images using text queries.

Default model: ViT-B/32 (512 dimensions)
"""

from typing import List, Union, Optional
from pathlib import Path
import numpy as np

from .base_embedder import BaseEmbedder


class ImageEmbedder(BaseEmbedder):
    """
    Image embedder using OpenAI CLIP.
    
    CLIP (Contrastive Language-Image Pre-training) embeds both images
    and text into the same vector space, enabling cross-modal search.
    
    Attributes:
        model_name: CLIP model variant (e.g., 'ViT-B/32', 'ViT-L/14')
        device: Compute device ('cpu' or 'cuda')
        
    Example:
        >>> embedder = ImageEmbedder()
        >>> 
        >>> # Encode images
        >>> img_embs = embedder.encode(["path/to/image.jpg"], input_type="image")
        >>> 
        >>> # Encode text queries
        >>> txt_embs = embedder.encode(["a photo of a dog"], input_type="text")
        >>> 
        >>> # Now you can compute similarity between img_embs and txt_embs
    """
    
    # Model dimensions for CLIP variants
    MODEL_DIMENSIONS = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
        "RN50": 1024,
        "RN101": 512,
    }
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None
    ):
        """
        Initialize the image embedder.
        
        Args:
            model_name: CLIP model variant
            device: Compute device (None for auto-detection)
        """
        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        super().__init__(model_name, device)
        self._preprocess = None
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 512)
    
    def load_model(self) -> None:
        """Load the CLIP model."""
        if self._model is not None:
            return
        
        import clip
        
        print(f"Loading CLIP model: {self.model_name}...")
        self._model, self._preprocess = clip.load(
            self.model_name,
            device=self.device
        )
        self._model.eval()
        print(f"  Model loaded on {self.device}. Dimension: {self._dimension}")
    
    def encode(
        self,
        inputs: Union[str, Path, List[Union[str, Path]]],
        batch_size: int = 16,
        show_progress: bool = True,
        normalize: bool = True,
        input_type: str = "auto"
    ) -> np.ndarray:
        """
        Encode inputs into CLIP embeddings.
        
        This method can encode both images and text queries.
        
        Args:
            inputs: Image paths OR text queries
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings
            input_type: 'text', 'image', or 'auto' (auto-detect)
            
        Returns:
            numpy array of shape (n_inputs, embedding_dim)
        """
        # Ensure model is loaded
        self.load_model()
        
        # Handle single input
        if isinstance(inputs, (str, Path)):
            inputs = [inputs]
        
        # Auto-detect input type
        if input_type == "auto":
            first = inputs[0]
            if isinstance(first, Path):
                input_type = "image"
            elif isinstance(first, str) and Path(first).exists():
                input_type = "image"
            else:
                input_type = "text"
        
        # Route to appropriate encoder
        if input_type == "text":
            return self._encode_text(inputs, normalize)  # type: ignore
        else:
            return self._encode_images(inputs, normalize, show_progress)  # type: ignore
    
    def _encode_text(
        self,
        texts: List[str],
        normalize: bool
    ) -> np.ndarray:
        """Encode text queries into CLIP embeddings."""
        import torch
        import clip
        
        with torch.no_grad():
            # Tokenize and encode
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            embeddings = self._model.encode_text(tokens)
            
            # Normalize if requested
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            return embeddings.cpu().numpy().astype(np.float32)
    
    def _encode_images(
        self,
        image_paths: List[Union[str, Path]],
        normalize: bool,
        show_progress: bool
    ) -> np.ndarray:
        """Encode images into CLIP embeddings."""
        import torch
        from PIL import Image
        from tqdm import tqdm
        
        embeddings = []
        
        # Create iterator with optional progress bar
        iterator = image_paths
        if show_progress:
            iterator = tqdm(image_paths, desc="Encoding images")
        
        with torch.no_grad():
            for path in iterator:
                # Load and preprocess image
                image = Image.open(path).convert("RGB")
                image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
                
                # Encode
                emb = self._model.encode_image(image_tensor)
                
                # Normalize if requested
                if normalize:
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                
                embeddings.append(emb.cpu().numpy()[0])
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convenience method to encode text queries.
        
        Args:
            texts: Single text or list of text queries
            normalize: Whether to L2-normalize
            
        Returns:
            numpy array of embeddings
        """
        return self.encode(texts, input_type="text", normalize=normalize)
    
    def encode_images(
        self,
        image_paths: Union[str, Path, List[Union[str, Path]]],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Convenience method to encode images.
        
        Args:
            image_paths: Single path or list of image paths
            normalize: Whether to L2-normalize
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        return self.encode(
            image_paths,
            input_type="image",
            normalize=normalize,
            show_progress=show_progress
        )
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    @property
    def modality(self) -> str:
        """Return the modality type."""
        return "image"
