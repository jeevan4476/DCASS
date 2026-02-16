"""
Abstract Base Class for Embedders

All embedding generators must inherit from this class and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding generators.
    
    All embedders must implement the encode() method that converts
    inputs into vector embeddings.
    
    Attributes:
        model_name: Name/identifier of the embedding model
        device: Compute device ('cpu', 'cuda', etc.)
        
    Example:
        >>> embedder = TextEmbedder("all-MiniLM-L6-v2")
        >>> embeddings = embedder.encode(["hello world", "test sentence"])
        >>> print(embeddings.shape)
        (2, 384)
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name/path of the embedding model
            device: Compute device ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the embedding model into memory.
        
        This is called lazily on first encode() call.
        Override this method to load your specific model.
        """
        pass
    
    @abstractmethod
    def encode(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode inputs into vector embeddings.
        
        Args:
            inputs: Single input or list of inputs to encode
            batch_size: Batch size for encoding (affects memory usage)
            show_progress: Whether to show a progress bar
            
        Returns:
            numpy array of shape (n_inputs, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        
        Returns:
            Integer dimension of output embeddings
        """
        pass
    
    @property
    @abstractmethod
    def modality(self) -> str:
        """
        Return the modality this embedder handles.
        
        Returns:
            One of: 'text', 'image', 'audio'
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
