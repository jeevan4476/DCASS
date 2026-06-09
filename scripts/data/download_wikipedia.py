#!/usr/bin/env python3
"""
Download Wikipedia Sentences for DCASS

Downloads and processes Wikipedia sentences for semantic indexing.
Uses the HuggingFace datasets library for easy access.

Usage:
    python scripts/download_wikipedia.py
    python scripts/download_wikipedia.py --num-sentences 100000
    python scripts/download_wikipedia.py --output data/raw/wikipedia
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Iterator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def clean_sentence(text: str) -> Optional[str]:
    """
    Clean and validate a sentence for indexing.
    
    Args:
        text: Raw sentence text
        
    Returns:
        Cleaned sentence or None if invalid
    """
    if not text:
        return None
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Skip very short or very long sentences
    if len(text) < 20 or len(text) > 300:
        return None
    
    # Skip sentences with too many special characters
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return None
    
    # Skip sentences that are likely headers, lists, or references
    skip_patterns = [
        r'^\d+\.',  # Numbered lists
        r'^[A-Z][a-z]+:',  # Headers like "Category:"
        r'\[\d+\]',  # References [1], [2]
        r'^See also',
        r'^References',
        r'^External links',
        r'^\|',  # Table rows
        r'^thumb\|',  # Image captions
        r'http[s]?://',  # URLs
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, text):
            return None
    
    # Ensure sentence ends properly
    if not text[-1] in '.!?"\'':
        return None
    
    return text


def extract_sentences_from_text(text: str) -> Iterator[str]:
    """
    Extract clean sentences from a text block.
    
    Args:
        text: Raw text block
        
    Yields:
        Clean sentences
    """
    # Simple sentence splitting (handles most cases)
    # Split on period followed by space and capital letter
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    for sent in raw_sentences:
        cleaned = clean_sentence(sent.strip())
        if cleaned:
            yield cleaned


def download_wikipedia_sentences(
    num_sentences: int = 50000,
    output_dir: Path = Path("data/raw/wikipedia"),
    min_article_length: int = 500,
) -> List[dict]:
    """
    Download and process Wikipedia sentences.
    
    Args:
        num_sentences: Target number of sentences
        output_dir: Directory to save sentences
        min_article_length: Minimum article length to consider
        
    Returns:
        List of sentence dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset
    
    print(f"\nDownloading Wikipedia dataset...")
    print(f"Target: {num_sentences:,} sentences")
    
    # Load Wikipedia dataset (streaming to avoid downloading entire thing)
    # Using wikimedia/wikipedia which is the new standard format
    dataset = None
    
    # Try different Wikipedia sources in order of preference
    sources = [
        ("wikimedia/wikipedia", "20231101.simple"),  # Simple English - cleaner, smaller
        ("wikimedia/wikipedia", "20231101.en"),      # Full English Wikipedia
        ("graelo/wikipedia", "20230901.en"),         # Alternative source
    ]
    
    for source, config in sources:
        try:
            print(f"  Trying {source} ({config})...")
            dataset = load_dataset(
                source,
                config,
                split="train",
                streaming=True,
            )
            print(f"  Successfully loaded {source}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    if dataset is None:
        raise RuntimeError("Could not load any Wikipedia dataset. Check your internet connection.")
    
    sentences = []
    articles_processed = 0
    
    print("\nProcessing articles...")
    
    for article in dataset:
        text = article.get("text", "")
        title = article.get("title", "Unknown")
        
        # Skip short articles
        if len(text) < min_article_length:
            continue
        
        # Extract sentences
        for sent in extract_sentences_from_text(text):
            sentences.append({
                "id": f"wiki_{len(sentences):06d}",
                "text": sent,
                "content": sent,  # For compatibility with index
                "source": "wikipedia",
                "article": title,
                "modality": "text",
            })
            
            if len(sentences) >= num_sentences:
                break
        
        articles_processed += 1
        
        if len(sentences) >= num_sentences:
            break
        
        # Progress update
        if articles_processed % 500 == 0:
            print(f"  Processed {articles_processed} articles, {len(sentences):,} sentences...")
    
    print(f"\nExtracted {len(sentences):,} sentences from {articles_processed} articles")
    
    # Save to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "sentences.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")
    
    # Also save a simple text file for inspection
    text_file = output_dir / "sentences.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for s in sentences[:1000]:  # First 1000 for inspection
            f.write(s["text"] + "\n")
    
    print(f"Sample saved to {text_file}")
    
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia sentences for DCASS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--num-sentences", "-n",
        type=int,
        default=50000,
        help="Number of sentences to download (default: 50000)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("storage/data/raw/wikipedia"),
        help="Output directory (default: data/raw/wikipedia)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DCASS Wikipedia Sentence Downloader")
    print("=" * 60)
    
    sentences = download_wikipedia_sentences(
        num_sentences=args.num_sentences,
        output_dir=args.output,
    )
    
    # Show sample
    print("\nSample sentences:")
    print("-" * 40)
    for s in sentences[:5]:
        print(f"  - {s['text'][:80]}...")
    
    print("\n" + "=" * 60)
    print(f"Download complete! {len(sentences):,} sentences saved.")
    print(f"\nNext step: Rebuild indices with:")
    print(f"  python scripts/build_indices.py --include-wikipedia")
    print("=" * 60)


if __name__ == "__main__":
    main()
