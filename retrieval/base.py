"""
Base Retriever Abstract Class

All retrieval modules must inherit from this base class and implement the search method.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SourceType(str, Enum):
    """Retrieval source types"""
    LOCAL_DENSE = "local_dense"
    BM25 = "bm25"
    EXTERNAL_KG = "external_kg"
    DEPMAP_KB = "depmap_kb"
    GRAPH_RAG = "graph_rag"
    WEB_SEARCH = "web_search"
    PUBMED = "pubmed"


@dataclass
class RetrievalResult:
    """Standardized retrieval result"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: SourceType
    rank: Optional[int] = None  # Set by fusion algorithm

    @property
    def source_type(self) -> SourceType:
        """Alias for source (for backwards compatibility)"""
        return self.source

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source.value,
            "rank": self.rank
        }


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval modules.

    All retrievers must implement:
    - search(): Async search interface
    - name property: Human-readable name
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable retriever name"""
        pass

    def __init__(self, source: Optional[SourceType] = None):
        """
        Initialize base retriever.

        Args:
            source: Optional default source type used by subclasses.
        """
        self._source_type = source

    @property
    def source_type(self) -> SourceType:
        """
        Source type identifier.

        Subclasses can override this property. If they do not, a source must be
        provided in `__init__`.
        """
        if self._source_type is None:
            raise NotImplementedError("Retriever must define source_type or pass source in __init__")
        return self._source_type

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Async search interface.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            **kwargs: Additional retriever-specific parameters

        Returns:
            List of RetrievalResult objects, sorted by score (descending)
        """
        pass

    async def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Batch search multiple queries.

        Args:
            queries: List of query strings
            top_k: Maximum results per query
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping query -> results
        """
        import asyncio

        # Execute searches in parallel
        tasks = [self.search(q, top_k, **kwargs) for q in queries]
        results = await asyncio.gather(*tasks)

        return {q: r for q, r in zip(queries, results)}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
