"""DCASS Corpus Loaders Package."""

from .base_loader import BaseLoader
from .flickr_loader import FlickrLoader
from .wikipedia_loader import WikipediaLoader

__all__ = [
    "BaseLoader",
    "FlickrLoader",
    "WikipediaLoader",
]
