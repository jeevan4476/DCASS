"""
Text Embedder using Sentence-Transformers

Generates dense vector embeddings for text using the Sentence-Transformers library.
Default model: all-MiniLM-L6-v2 (384 dimensions, fast and efficient)
"""

from typing import List, Union
import numpy as np

from .base_embedder import BaseEmbedder


class TextEmbedder(BaseEmbedder):
    """
    Text embedder using Sentence-Transformers.
    
    Generates semantic embeddings for text that capture meaning,
    allowing for semantic similarity search.
    
    Attributes:
        model_name: Name of the Sentence-Transformers model
        device: Compute device ('cpu' or 'cuda')
        
    Example:
        >>> embedder = TextEmbedder()
        >>> embeddings = embedder.encode(["Hello world", "How are you?"])
        >>> print(embeddings.shape)
        (2, 384)
    """
    
    # Model dimensions for common models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the Sentence-Transformers model
            device: Compute device ('cpu' or 'cuda')
        """
        super().__init__(model_name, device)
        self._dimension = self.MODEL_DIMENSIONS.get(model_name)
    
    def load_model(self) -> None:
        """Load the Sentence-Transformer model."""
        if self._model is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading text model: {self.model_name}...")
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device
        )
        
        # Get actual dimension from model
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"  Model loaded. Dimension: {self._dimension}")
    
    def encode(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text inputs into embeddings.
        
        Args:
            inputs: Single text string or list of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            
        Returns:
            numpy array of shape (n_inputs, embedding_dim)
        """
        # Ensure model is loaded
        self.load_model()
        
        # Handle single string input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Generate embeddings
        embeddings = self._model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            self.load_model()
        return self._dimension  # type: ignore
    
    @property
    def modality(self) -> str:
        """Return the modality type."""
        return "text"
