# src/corpus/embedders/clip_embedder.py
"""
CLIP-based embedder for images and text.

Uses OpenAI's CLIP (ViT-B/32) model to generate 512-dimensional embeddings
that are compatible across image and text modalities.

This enables cross-modal search:
- Text query -> Image results
- Text query -> Text results (captions)
- Image query -> Text results

Architecture:
    Input (image/text) -> CLIP Encoder -> L2 Normalize -> 512-dim embedding
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Optional
from PIL import Image

import torch
import clip


class CLIPEmbedder:
    """
    CLIP-based embedder for cross-modal semantic search.
    
    Supports embedding both images and text into a shared 512-dimensional
    vector space, enabling cross-modal similarity search.
    
    Usage:
        embedder = CLIPEmbedder()
        
        # Embed text
        text_emb = embedder.embed_text("a dog running on the beach")
        
        # Embed image
        img_emb = embedder.embed_image("path/to/image.jpg")
        
        # Batch embedding
        embeddings = embedder.embed_texts(["query1", "query2", "query3"])
    
    Attributes:
        model_name: CLIP model variant (default: "ViT-B/32")
        device: Compute device ('cuda' or 'cpu')
        embedding_dim: Output embedding dimension (512 for ViT-B/32)
    """
    
    MODEL_NAME = "ViT-B/32"
    EMBEDDING_DIM = 512
    
    def __init__(self, device: str = None):
        """
        Initialize the CLIP embedder.
        
        Args:
            device: Compute device. Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load the CLIP model."""
        if not self._loaded:
            print(f"Loading CLIP model ({self.MODEL_NAME}) on {self.device}...")
            self._model, self._preprocess = clip.load(self.MODEL_NAME, device=self.device)
            self._model.eval()
            self._loaded = True
    
    @property
    def model(self):
        """Get the CLIP model (loads if needed)."""
        self._ensure_loaded()
        return self._model
    
    @property
    def preprocess(self):
        """Get the image preprocessing function."""
        self._ensure_loaded()
        return self._preprocess
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized 512-dim embedding as numpy array
        """
        self._ensure_loaded()
        
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype("float32").squeeze()
    
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of shape (n_texts, 512) with normalized embeddings
        """
        self._ensure_loaded()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = clip.tokenize(batch, truncate=True).to(self.device)
                embeddings = self._model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype("float32")
    
    def embed_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Embed a single image.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Normalized 512-dim embedding as numpy array
        """
        self._ensure_loaded()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        with torch.no_grad():
            processed = self._preprocess(image).unsqueeze(0).to(self.device)
            embedding = self._model.encode_image(processed)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype("float32").squeeze()
    
    def embed_images(
        self,
        images: list[Union[str, Path, Image.Image]],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Embed multiple images in batches.
        
        Args:
            images: List of image paths or PIL Image objects
            batch_size: Batch size for processing
            
        Returns:
            Array of shape (n_images, 512) with normalized embeddings
        """
        self._ensure_loaded()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Load and preprocess images
                processed_batch = []
                for img in batch:
                    if isinstance(img, (str, Path)):
                        img = Image.open(img).convert("RGB")
                    processed_batch.append(self._preprocess(img))
                
                # Stack and encode
                batch_tensor = torch.stack(processed_batch).to(self.device)
                embeddings = self._model.encode_image(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype("float32")
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        target_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and targets.
        
        Args:
            query_embedding: Single embedding (512,) or batch (n, 512)
            target_embeddings: Target embeddings (m, 512)
            
        Returns:
            Similarity scores
        """
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity (embeddings are already normalized)
        return np.dot(query_embedding, target_embeddings.T)
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"CLIPEmbedder(model={self.MODEL_NAME}, device={self.device}, {status})"
