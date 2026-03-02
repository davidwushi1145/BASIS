"""
Dense Retriever using Qwen3-Embedding Model

Loads local JSONL corpus and performs vector similarity search.
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from pathlib import Path

from .base import BaseRetriever, RetrievalResult, SourceType
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using pre-computed embeddings.

    Loads corpus from JSONL file and uses cached vectors for fast search.
    """

    def __init__(
        self,
        embedder: Any,  # QwenEmbedder instance
        corpus_file: Optional[str] = None,
        vector_cache: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Dense Retriever.

        Args:
            embedder: QwenEmbedder instance for query embedding
            corpus_file: Path to corpus JSONL file
            vector_cache: Path to precomputed vector cache (.pt)
            device: Torch device (cuda:0, cpu, etc.)
        """
        self.embedder = embedder
        self.corpus_file = corpus_file or settings.CORPUS_FILE
        self.vector_cache = vector_cache or settings.vector_cache_path
        self.device = device or settings.DEVICE

        # Data storage
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.doc_embeddings: Optional[torch.Tensor] = None

        # Load corpus on initialization
        self._load_corpus()

    @property
    def name(self) -> str:
        return "Dense Retriever (Qwen Embeddings)"

    @property
    def source_type(self) -> SourceType:
        return SourceType.LOCAL_DENSE

    def _load_corpus(self):
        """Load corpus and precomputed vectors"""
        logger.info(
            "Loading corpus",
            corpus_file=self.corpus_file,
            vector_cache=self.vector_cache
        )

        # Load documents and metadata
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.documents.append(item['text'])
                self.metadatas.append(item.get('metadata', {}))

        # Load precomputed vectors
        if not os.path.exists(self.vector_cache):
            raise FileNotFoundError(
                f"Vector cache not found: {self.vector_cache}. "
                f"Run scripts/precompute_vectors.py first."
            )

        self.doc_embeddings = torch.load(
            self.vector_cache,
            map_location=self.device
        )

        logger.info(
            "Corpus loaded successfully",
            num_documents=len(self.documents),
            embedding_shape=self.doc_embeddings.shape
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search corpus using dense vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Ignored

        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        # Embed query
        query_embedding = await self.embedder.embed_async(query)
        query_tensor = torch.tensor(
            query_embedding,
            device=self.device
        ).unsqueeze(0)  # [1, dim]

        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_tensor,
            self.doc_embeddings,
            dim=1
        )  # [num_docs]

        # Get top-k indices
        top_scores, top_indices = torch.topk(
            similarities,
            k=min(top_k, len(self.documents))
        )

        # Convert to RetrievalResult objects
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append(
                RetrievalResult(
                    content=self.documents[idx],
                    metadata=self.metadatas[idx],
                    score=score,
                    source=self.source_type
                )
            )

        logger.debug(
            "Dense search completed",
            query=query[:50],
            num_results=len(results),
            top_score=results[0].score if results else 0.0
        )

        return results

    def get_document_count(self) -> int:
        """Get total number of documents in corpus"""
        return len(self.documents)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0
