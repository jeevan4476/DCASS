"""
Abstract Base Class for Corpus Loaders

All data loaders must inherit from this class and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Dict, Any, List


class BaseLoader(ABC):
    """
    Abstract base class for loading corpus data.

    All loaders must implement the load() method that yields
    individual items with their metadata.

    Attributes:
        source_path: Path to the data source (file or directory)

    Example:
        >>> loader = FlickrLoader("data/raw/flickr8k")
        >>> for item in loader.load():
        ...     print(item["id"], item["content"])
    """

    def __init__(self, source_path: Path):
        """
        Initialize the loader with a source path.

        Args:
            source_path: Path to the data source

        Raises:
            FileNotFoundError: If source path doesn't exist
        """
        self.source_path = Path(source_path)
        self._validate_source()

    def _validate_source(self) -> None:
        """Validate that the source path exists."""
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source path not found: {self.source_path}"
            )

    @abstractmethod
    def load(self) -> Iterator[Dict[str, Any]]:
        """
        Yield items from the corpus.

        Each item should be a dictionary with at least:
            - 'id': Unique identifier for this item
            - 'content': The actual content (text string, image path, etc.)
            - 'metadata': Additional metadata dictionary

        Yields:
            Dict containing id, content, and metadata
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return total number of items in the corpus.

        Returns:
            Number of items
        """
        pass

    @property
    @abstractmethod
    def modality(self) -> str:
        """
        Return the modality type.

        Returns:
            One of: 'text', 'image', 'audio'
        """
        pass

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Load all items into memory.

        Warning: May consume significant memory for large corpora.

        Returns:
            List of all items
        """
        return list(self.load())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source_path})"
