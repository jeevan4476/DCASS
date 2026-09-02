"""
Tests for Google Drive URL and relative path resolution on MediaItem.
"""
from src.corpus.index.unified_index import MediaItem


def test_media_item_gdrive_path_image():
    item_8k = MediaItem(
        id="flickr8k_12345",
        modality="image",
        content="path/to/img.jpg",
        score=0.9,
        normalized_score=0.9,
        metadata={"source": "flickr8k", "path": "flickr8k/images/12345.jpg"}
    )
    assert item_8k.gdrive_path == "data/raw/flickr8k/images/12345.jpg"
    assert "https://drive.google.com/drive/u/0/search?q=" in item_8k.gdrive_url

    item_30k = MediaItem(
        id="flickr30k_67890",
        modality="image",
        content="path/to/img.jpg",
        score=0.9,
        normalized_score=0.9,
        metadata={"source": "flickr30k", "path": "storage/data/raw/flickr30k/images/67890.jpg"}
    )
    assert item_30k.gdrive_path == "data/raw/flickr30k/images/67890.jpg"


def test_media_item_gdrive_path_audio():
    item_aud = MediaItem(
        id="audio_000001",
        modality="audio",
        content="speech",
        score=0.8,
        normalized_score=0.8,
        metadata={"audio_path": "audio/cache/audio_000001.wav"}
    )
    assert item_aud.gdrive_path == "data/audio/audio_000001.wav"
    assert "audio_000001.wav" in item_aud.gdrive_url


def test_media_item_gdrive_path_text():
    item_txt = MediaItem(
        id="wiki_000042",
        modality="text",
        content="Wikipedia sentence",
        score=0.85,
        normalized_score=0.85,
        metadata={"source": "wikipedia"}
    )
    assert item_txt.gdrive_path == "data/raw/wikipedia/sentences.json"
    assert "sentences.json" in item_txt.gdrive_url


def test_media_item_custom_gdrive_base_url(monkeypatch):
    monkeypatch.setenv("DCASS_GDRIVE_BASE_URL", "https://drive.google.com/drive/folders/1ABCXYZ")
    item = MediaItem(
        id="audio_000001",
        modality="audio",
        content="speech",
        score=0.8,
        normalized_score=0.8,
        metadata={"audio_path": "audio/cache/audio_000001.wav"}
    )
    assert item.gdrive_url == "https://drive.google.com/drive/folders/1ABCXYZ/data/audio/audio_000001.wav"
