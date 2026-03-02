"""
Precompute Vector Cache

Generates embeddings for corpus and saves to cache file.
Run this before starting the server for faster loading.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm

from config import settings
from models.embedder import QwenEmbedder
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Precompute vector cache"""
    logger.info(
        "Starting vector precomputation",
        corpus_file=settings.CORPUS_FILE,
        output_file=settings.vector_cache_path
    )

    # Check if corpus exists
    corpus_path = Path(settings.CORPUS_FILE)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {settings.CORPUS_FILE}")

    # Load embedder
    logger.info("Loading embedder model", path=settings.EMBEDDING_PATH)
    embedder = QwenEmbedder(
        model_path=settings.EMBEDDING_PATH,
        device=settings.DEVICE
    )

    # Load documents
    logger.info("Loading documents")
    documents = []
    with open(settings.CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            documents.append(item['text'])

    logger.info(f"Loaded {len(documents)} documents")

    # Compute embeddings with progress bar
    logger.info("Computing embeddings (this may take a while)...")
    embeddings = []

    batch_size = 32
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
        batch = documents[i:i + batch_size]
        batch_embeddings = embedder.embed_batch(batch)
        embeddings.extend(batch_embeddings)

    # Convert to tensor
    embedding_tensor = torch.stack([torch.tensor(e) for e in embeddings])

    # Save cache
    output_path = Path(settings.vector_cache_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving cache to {output_path}")
    torch.save(embedding_tensor, output_path)

    logger.info(
        "Vector cache created successfully",
        shape=embedding_tensor.shape,
        file_size_mb=output_path.stat().st_size / 1024 / 1024
    )


if __name__ == "__main__":
    main()
