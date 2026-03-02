"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF)

Combines multiple retrieval sources using RRF algorithm and applies
biomedical entity boosting with fine-grained weighting (matching original implementation).
"""

import asyncio
import re
import hashlib
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from .base import BaseRetriever, RetrievalResult, SourceType
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that fuses results from multiple sources.

    Implements:
    - Reciprocal Rank Fusion (RRF)
    - HyDE (Hypothetical Document Embeddings)
    - NER-based entity boosting
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: Optional[BaseRetriever] = None,
        external_kg_retriever: Optional[BaseRetriever] = None,
        hyde_generator: Optional[Any] = None,
        entity_analyzer: Optional[Any] = None,
        rrf_k: int = 60
    ):
        """
        Initialize Hybrid Retriever.

        Args:
            dense_retriever: Dense vector retriever
            sparse_retriever: BM25 retriever (optional)
            external_kg_retriever: External KG retriever (optional)
            hyde_generator: HyDE document generator (optional)
            entity_analyzer: NER entity analyzer (optional)
            rrf_k: RRF constant (default: 60)
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.external_kg = external_kg_retriever
        self.hyde_generator = hyde_generator
        self.entity_analyzer = entity_analyzer
        self.rrf_k = rrf_k

        # Active retrievers
        self.retrievers = [r for r in [dense_retriever, sparse_retriever, external_kg_retriever] if r]

        logger.info(
            "HybridRetriever initialized",
            num_retrievers=len(self.retrievers),
            use_hyde=hyde_generator is not None,
            use_entity_boost=entity_analyzer is not None
        )

    @property
    def name(self) -> str:
        return "Hybrid Retriever (RRF Fusion)"

    @property
    def source_type(self) -> SourceType:
        return SourceType.LOCAL_DENSE  # Primary source

    async def search(
        self,
        query: str,
        top_k: int = 20,
        use_hyde: Optional[bool] = None,
        use_entity_boost: Optional[bool] = None,
        weights: Optional[Dict[SourceType, float]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Hybrid search with RRF fusion.

        Args:
            query: Search query
            top_k: Number of final results
            use_hyde: Enable HyDE (default: from settings)
            use_entity_boost: Enable entity boosting (default: from settings)
            weights: Source-specific weights for RRF
            **kwargs: Additional parameters

        Returns:
            Fused and ranked results
        """
        use_hyde = use_hyde if use_hyde is not None else settings.USE_HYDE
        use_entity_boost = use_entity_boost if use_entity_boost is not None else settings.USE_ENTITY_BOOST

        # Step 1: Generate HyDE document if enabled.
        # Orchestrator may pass a pre-generated HyDE doc via kwargs.
        hyde_doc = kwargs.pop("hyde_doc", None)
        if hyde_doc:
            hyde_doc = hyde_doc.strip()

        if use_hyde and not hyde_doc and self.hyde_generator:
            try:
                hyde_doc = await self.hyde_generator.generate(query)
                logger.debug("HyDE document generated", length=len(hyde_doc) if hyde_doc else 0)
            except Exception as e:
                logger.warning("HyDE generation failed", error=str(e))

        # Step 2: Execute all retrievers in parallel
        tasks = []
        task_labels = []

        # Dense retrieval (original query)
        tasks.append(self.dense.search(query, top_k * 2))
        task_labels.append(("dense", SourceType.LOCAL_DENSE))

        # Dense retrieval (HyDE query)
        if hyde_doc:
            tasks.append(self.dense.search(hyde_doc, top_k * 2))
            task_labels.append(("dense_hyde", SourceType.LOCAL_DENSE))

        # Sparse retrieval
        if self.sparse:
            tasks.append(self.sparse.search(query, top_k * 2))
            task_labels.append(("sparse", SourceType.BM25))

        # External KG retrieval
        if self.external_kg:
            tasks.append(self.external_kg.search(query, top_k))
            task_labels.append(("external_kg", SourceType.EXTERNAL_KG))

        # Execute in parallel
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                logger.error(
                    "Retriever failed",
                    retriever=task_labels[i][0],
                    error=str(result)
                )
            else:
                valid_results.append((task_labels[i], result))

        # Step 3: Apply RRF fusion
        if weights is None:
            # ev1 parity: local fusion uses unweighted RRF (equal contribution).
            weights = {
                SourceType.LOCAL_DENSE: 1.0,
                SourceType.BM25: 1.0,
                SourceType.EXTERNAL_KG: 1.0,
                "dense_hyde": 1.0,
            }

        fused_results = self._rrf_fusion(valid_results, weights)

        # Step 4: Apply entity boosting
        if use_entity_boost and self.entity_analyzer:
            fused_results = await self._apply_entity_boost(query, fused_results)

        # Step 5: Return top-k
        return fused_results[:top_k]

    def _rrf_fusion(
        self,
        results_list: List[tuple],
        weights: Dict[Any, float]
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion algorithm.

        RRF formula: score(d) = Σ (weight_i / (k + rank_i))

        Args:
            results_list: List of (label, results) tuples
            weights: Source-specific weights

        Returns:
            Fused results sorted by RRF score
        """
        rrf_scores = defaultdict(float)
        doc_objects = {}  # Store first occurrence of each document

        for (label, source_type), results in results_list:
            # Label-specific weights (e.g. dense_hyde) override source-level weights.
            weight = weights.get(label, weights.get(source_type, 1.0))

            for rank, result in enumerate(results, start=1):
                # Use content as unique key (hash for efficiency)
                doc_key = hash(result.content)

                # RRF score contribution
                rrf_scores[doc_key] += weight / (self.rrf_k + rank)

                # Store document (first occurrence)
                if doc_key not in doc_objects:
                    doc_objects[doc_key] = result

        # Sort by RRF score (descending)
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # Build final results with updated scores
        fused_results = []
        for rank, doc_key in enumerate(sorted_keys, start=1):
            result = doc_objects[doc_key]
            result.score = rrf_scores[doc_key]  # Update to RRF score
            result.rank = rank
            fused_results.append(result)

        logger.debug(
            "RRF fusion completed",
            num_sources=len(results_list),
            total_unique_docs=len(fused_results),
            top_score=fused_results[0].score if fused_results else 0.0
        )

        return fused_results

    async def _apply_entity_boost(
        self,
        query: str,
        results: List[RetrievalResult],
        query_entities: Optional[Dict[str, Set[str]]] = None
    ) -> List[RetrievalResult]:
        """
        Fine-grained entity boosting matching original implementation.

        Boost levels:
        - Title contains "synthetic lethal" or "collateral lethality": +0.3
        - Document contains query genes: +0.15
        - Document contains query keywords: +0.05
        - Web source: +0.05

        Args:
            query: Original query
            results: RRF-fused results
            query_entities: Pre-extracted entities (optional)

        Returns:
            Boosted and re-sorted results
        """
        # Extract entities from query if not provided
        if query_entities is None:
            if self.entity_analyzer:
                query_entities = await self.entity_analyzer.extract(query)
            else:
                query_entities = {"genes": set(), "keywords": set()}

        gene_entities = set(query_entities.get('genes', []))
        keyword_entities = set(query_entities.get('keywords', []))

        if not gene_entities and not keyword_entities:
            logger.debug("No entities found for boosting")
            return results

        logger.debug(
            "Applying fine-grained entity boost",
            genes=list(gene_entities)[:5],
            keywords=list(keyword_entities)[:5]
        )

        # Apply boosting to each result
        for result in results:
            boost = 0.0
            metadata = result.metadata or {}
            title = metadata.get('title', metadata.get('paper_title', '')).lower()

            # Boost 1: Title contains core SL concepts (highest boost: +0.3)
            if 'collateral lethality' in title or 'synthetic lethal' in title:
                boost += 0.3

            # Boost 2: Document contains query genes (+0.15)
            chunk_genes = set(metadata.get('key_genes', []))
            if gene_entities & chunk_genes:
                boost += 0.15

            # Boost 3: Document contains query keywords (+0.05)
            chunk_methods = set(metadata.get('key_methods', []))
            if keyword_entities & chunk_methods:
                boost += 0.05

            # Boost 4: Web source bonus (+0.05)
            if metadata.get('is_web') or result.source == SourceType.WEB_SEARCH:
                boost += 0.05

            # Apply multiplicative boost: score * (1 + boost)
            if boost > 0:
                result.score = result.score * (1 + boost)
                result.metadata['entity_boost'] = boost

        # Re-sort by boosted scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(results, start=1):
            result.rank = rank

        return results

    def diversified_retrieval(
        self,
        candidates: List[RetrievalResult],
        min_papers: int = None,
        max_per_paper: int = None,
        score_threshold: float = None
    ) -> List[RetrievalResult]:
        """
        Round-Robin diversity selection based on paper title fingerprint.

        Prevents the same paper from appearing multiple times across different
        websites (PubMed, Nature, Cell, etc.) by using title-based fingerprinting.

        Args:
            candidates: Scored candidate results
            min_papers: Minimum number of unique papers (default: from settings)
            max_per_paper: Maximum chunks per paper (default: from settings)
            score_threshold: Minimum relevance score (default: from settings)

        Returns:
            Diversified results
        """
        min_papers = min_papers or settings.MIN_PAPERS
        max_per_paper = max_per_paper or settings.MAX_CHUNKS_PER_PAPER
        score_threshold = score_threshold or settings.SCORE_THRESHOLD

        # Filter by score threshold
        qualified_candidates = [
            c for c in candidates if c.score >= score_threshold
        ]

        if not qualified_candidates:
            logger.debug("No candidates above score threshold", threshold=score_threshold)
            return []

        # Group by paper title fingerprint
        groups: Dict[str, List[RetrievalResult]] = defaultdict(list)

        for cand in qualified_candidates:
            metadata = cand.metadata or {}
            title = metadata.get('title', metadata.get('paper_title', '')).strip()
            link = metadata.get('link', metadata.get('url', ''))
            source_id = metadata.get('source', '')

            # Build fingerprint: prefer cleaned title, fallback to URL or source ID
            clean_title = re.sub(r'[^\w\s]', '', title).lower()

            if len(clean_title) > 15:
                # Use first 60 chars of title as fingerprint (prevents minor title variations)
                group_key = clean_title[:60]
            elif link:
                # Use URL domain + path as fingerprint
                group_key = link
            elif source_id:
                # Match ev1: local results use an explicit local_ prefix to avoid collisions.
                group_key = f"local_{source_id}"
            else:
                # Fallback: stable hash (matches ev1's md5 of first ~200 chars).
                digest = hashlib.md5((cand.content or "")[:200].encode("utf-8", errors="ignore")).hexdigest()[:16]
                group_key = f"hash_{digest}"

            groups[group_key].append(cand)

        logger.debug(
            "Diversity grouping completed",
            total_candidates=len(qualified_candidates),
            unique_groups=len(groups)
        )

        # Round-Robin selection (ev1 parity)
        selected: List[RetrievalResult] = []
        group_keys = list(groups.keys())
        selection_counts: Dict[str, int] = defaultdict(int)
        round_index = 0
        max_total_results = 15

        while len(selected) < max_total_results and any(groups.values()):
            current_key = group_keys[round_index % len(group_keys)]
            current_group_list = groups[current_key]

            if current_group_list:
                if selection_counts[current_key] < max_per_paper:
                    best_cand = max(current_group_list, key=lambda x: x.score)
                    selected.append(best_cand)
                    current_group_list.remove(best_cand)
                    selection_counts[current_key] += 1

            round_index += 1

            # Safety break: if we spin too long, we're out of selectable items.
            if round_index > len(group_keys) * max_per_paper * 2:
                break

        # Keep selection order to preserve diversity (do not re-sort).
        for rank, result in enumerate(selected, start=1):
            result.rank = rank

        logger.info(
            "Diversified retrieval completed",
            input_count=len(candidates),
            output_count=len(selected),
            unique_papers=len({self._get_fingerprint(r) for r in selected})
        )

        return selected

    def _get_fingerprint(self, result: RetrievalResult) -> str:
        """Get fingerprint for a result (for deduplication)."""
        metadata = result.metadata or {}
        title = metadata.get('title', metadata.get('paper_title', '')).strip()
        clean_title = re.sub(r'[^\w\s]', '', title).lower()

        if len(clean_title) > 15:
            return clean_title[:60]

        link = metadata.get('link', metadata.get('url', ''))
        if link:
            return link

        source_id = metadata.get('source', '')
        if source_id:
            return f"local_{source_id}"

        digest = hashlib.md5((result.content or "")[:200].encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"hash_{digest}"

    async def search_multi(
        self,
        queries: List[str],
        top_k: int = 20,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search multiple queries and merge results.

        Args:
            queries: List of query strings
            top_k: Total number of results to return
            **kwargs: Additional parameters

        Returns:
            Merged and deduplicated results
        """
        # Execute all queries in parallel.
        # If a pre-generated HyDE doc is provided, apply it only to the first
        # (primary/original) query to mirror ev1 behavior.
        hyde_doc = kwargs.pop("hyde_doc", None)
        tasks = []
        for i, q in enumerate(queries):
            task_kwargs = dict(kwargs)
            if hyde_doc:
                if i == 0:
                    task_kwargs["hyde_doc"] = hyde_doc
                    task_kwargs["use_hyde"] = True
                else:
                    task_kwargs["use_hyde"] = False
            tasks.append(self.search(q, top_k, **task_kwargs))
        results_per_query = await asyncio.gather(*tasks)

        # Merge and deduplicate
        seen_content = set()
        merged_results = []

        for results in results_per_query:
            for result in results:
                content_key = hash(result.content)
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    merged_results.append(result)

        # Sort by score
        merged_results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(merged_results[:top_k], start=1):
            result.rank = rank

        logger.info(
            "Multi-query search completed",
            num_queries=len(queries),
            total_results=len(merged_results),
            final_results=min(top_k, len(merged_results))
        )

        return merged_results[:top_k]
