"""
BM25 Sparse Retriever

Implements BM25 algorithm with biomedical-aware tokenization.
Preserves gene names as complete tokens (e.g., "TP53", "BRCA1").
"""

import re
from typing import Any, List, Optional, Set
from rank_bm25 import BM25Okapi

from .base import BaseRetriever, RetrievalResult, SourceType
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class SparseRetriever(BaseRetriever):
    """
    BM25-based sparse retrieval with biomedical NER-aware tokenization.

    Tokenization strategy:
    1. Preserve gene names (uppercase alphanumeric, 2-10 chars)
    2. Use NER model to extract biomedical entities
    3. Fall back to standard word tokenization
    """

    def __init__(
        self,
        documents: List[str],
        metadatas: List[dict],
        ner_pipeline: Optional[Any] = None
    ):
        """
        Initialize BM25 Retriever.

        Args:
            documents: Corpus documents
            metadatas: Document metadata
            ner_pipeline: Biomedical NER pipeline (optional)
        """
        self.documents = documents
        self.metadatas = metadatas
        self.ner_pipeline = ner_pipeline

        # Tokenize all documents
        logger.info("Tokenizing corpus for BM25", num_docs=len(documents))
        self.tokenized_corpus = [
            self._tokenize(doc) for doc in documents
        ]

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index created successfully")

    @property
    def name(self) -> str:
        return "BM25 Sparse Retriever"

    @property
    def source_type(self) -> SourceType:
        return SourceType.BM25

    def _tokenize(self, text: str) -> List[str]:
        """
        Biomedical-aware tokenization.

        Strategy (following original code):
        1. Extract gene names via regex (e.g., TP53, BRCA1)
        2. Extract NER entities if pipeline available
        3. Standard word tokenization
        4. Preserve gene names as complete tokens

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        tokens = []

        # Step 1: Rule-based gene extraction
        # Pattern: uppercase letter + alphanumeric (2-10 chars)
        # Avoids word boundaries to handle Chinese (e.g., "STK11缺失")
        gene_pattern = r'(?:^|[^A-Z0-9])([A-Z][A-Z0-9]{1,9})(?=$|[^A-Z0-9])'
        potential_genes = re.findall(gene_pattern, text.upper())

        # Filter blacklist
        gene_blacklist = {"AND", "NOT", "THE", "FOR", "WITH", "FROM"}
        genes = {
            g for g in potential_genes
            if len(g) >= 2 and g not in gene_blacklist
        }

        tokens.extend(genes)

        # Step 2: NER-based extraction (if available)
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                gene_like_labels = {
                    'Gene_or_gene_product',
                    'Gene',
                    'Protein',
                    'Coreference',
                    'Diagnostic_procedure'
                }

                for entity in ner_results:
                    if entity.get('entity_group') in gene_like_labels:
                        word = entity['word'].strip().replace('##', '')  # Clean BERT tokens
                        if len(word) > 1:
                            tokens.append(word.upper())
            except Exception as e:
                logger.warning("NER extraction failed", error=str(e))

        # Step 3: Standard tokenization
        # Split on whitespace and punctuation, lowercase
        words = re.findall(r'\b\w+\b', text.lower())
        tokens.extend(words)

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens

    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        BM25 sparse search.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of RetrievalResult
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("Query tokenization produced no tokens", query=query)
            return []

        # BM25 scoring
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices (argsort descending)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # Only include non-zero scores
                results.append(
                    RetrievalResult(
                        content=self.documents[idx],
                        metadata=self.metadatas[idx],
                        score=score,
                        source=self.source_type
                    )
                )

        logger.debug(
            "BM25 search completed",
            query_tokens=query_tokens[:10],
            num_results=len(results),
            top_score=results[0].score if results else 0.0
        )

        return results

    def get_avg_doc_len(self) -> float:
        """Get average document length (in tokens)"""
        return self.bm25.avgdl

    def get_num_docs(self) -> int:
        """Get number of documents"""
        return len(self.documents)
