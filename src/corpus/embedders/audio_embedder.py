# src/corpus/embedders/audio_embedder.py
"""
CLAP-based embedder for audio.

Uses LAION's CLAP (Contrastive Language-Audio Pretraining) model to generate
512-dimensional embeddings that are compatible with CLIP embeddings.

This enables:
- Text query -> Audio results
- Audio query -> Audio results
- Cross-modal search with images and text

Architecture:
    Input (audio/text) -> CLAP Encoder -> L2 Normalize -> 512-dim embedding

Note: CLAP and CLIP embeddings are in similar but not identical vector spaces.
      Score normalization is required for fair comparison.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union

import torch


class AudioEmbedder:
    """
    CLAP-based embedder for audio semantic search.

    Supports embedding both audio files and text into a shared 512-dimensional
    vector space, enabling text-to-audio and audio-to-audio similarity search.

    Usage:
        embedder = AudioEmbedder()

        # Embed text (for querying audio)
        text_emb = embedder.embed_text("a dog barking loudly")

        # Embed audio file
        audio_emb = embedder.embed_audio("path/to/audio.wav")

    Requirements:
        - transformers
        - librosa (for audio loading)

    Attributes:
        model_name: CLAP model variant
        device: Compute device ('cuda' or 'cpu')
        embedding_dim: Output embedding dimension (512)
        sample_rate: Expected audio sample rate (48000 Hz for CLAP)
    """

    MODEL_NAME = "laion/clap-htsat-unfused"
    EMBEDDING_DIM = 512
    SAMPLE_RATE = 48000

    def __init__(self, device: str = None):
        """
        Initialize the CLAP audio embedder.

        Args:
            device: Compute device. Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the CLAP model."""
        if not self._loaded:
            try:
                from transformers import ClapModel, ClapProcessor
            except ImportError:
                raise ImportError(
                    "CLAP requires 'transformers' package. "
                    "Install with: pip install transformers"
                )

            print(f"Loading CLAP model ({self.MODEL_NAME}) on {self.device}...")
            self._processor = ClapProcessor.from_pretrained(self.MODEL_NAME)
            self._model = ClapModel.from_pretrained(self.MODEL_NAME).to(self.device)
            self._model.eval()
            self._loaded = True

    @property
    def model(self):
        """Get the CLAP model (loads if needed)."""
        self._ensure_loaded()
        return self._model

    @property
    def processor(self):
        """Get the CLAP processor."""
        self._ensure_loaded()
        return self._processor

    def _load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file and resample to expected rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "Audio loading requires 'librosa' package. "
                "Install with: pip install librosa"
            )

        waveform, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        return waveform

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string for audio search.

        Args:
            text: Text description to embed

        Returns:
            Normalized 512-dim embedding as numpy array
        """
        self._ensure_loaded()

        with torch.no_grad():
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            embedding = self._model.get_text_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype("float32").squeeze()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of text descriptions
            batch_size: Batch size for processing

        Returns:
            Array of shape (n_texts, 512) with normalized embeddings
        """
        self._ensure_loaded()

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                embeddings = self._model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype("float32")

    def embed_audio(self, audio: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Embed a single audio file or waveform.

        Args:
            audio: Path to audio file or numpy waveform array

        Returns:
            Normalized 512-dim embedding as numpy array
        """
        self._ensure_loaded()

        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            waveform = self._load_audio(audio)
        else:
            waveform = audio

        with torch.no_grad():
            inputs = self._processor(
                audios=[waveform],
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            embedding = self._model.get_audio_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype("float32").squeeze()

    def embed_audios(
        self,
        audios: list[Union[str, Path, np.ndarray]],
        batch_size: int = 4
    ) -> np.ndarray:
        """
        Embed multiple audio files in batches.

        Args:
            audios: List of audio file paths or waveforms
            batch_size: Batch size (keep small for memory)

        Returns:
            Array of shape (n_audios, 512) with normalized embeddings
        """
        self._ensure_loaded()

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(audios), batch_size):
                batch = audios[i:i + batch_size]

                # Load waveforms
                waveforms = []
                for audio in batch:
                    if isinstance(audio, (str, Path)):
                        waveforms.append(self._load_audio(audio))
                    else:
                        waveforms.append(audio)

                inputs = self._processor(
                    audios=waveforms,
                    sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                embeddings = self._model.get_audio_features(**inputs)
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
        return f"AudioEmbedder(model={self.MODEL_NAME}, device={self.device}, {status})"
