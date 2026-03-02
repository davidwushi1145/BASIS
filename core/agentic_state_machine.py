"""
Agentic RAG State Machine

Implements multi-hop reasoning with step-back mechanism.
Automatically triggers when initial retrieval quality is below threshold.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import settings
from retrieval import RetrievalResult, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


class AgenticState(str, Enum):
    """States in the agentic reasoning loop"""
    INITIAL = "initial"
    STEP_BACK = "step_back"
    MECHANISTIC_SEARCH = "mechanistic_search"
    VERIFY = "verify"
    DONE = "done"


@dataclass
class AgenticContext:
    """Context for agentic reasoning loop"""
    query: str
    results: List[RetrievalResult]
    entities: Optional[Dict[str, Any]] = None
    hop: int = 0
    mechanistic_queries: List[str] = None
    state_history: List[AgenticState] = None

    def __post_init__(self):
        if self.mechanistic_queries is None:
            self.mechanistic_queries = []
        if self.state_history is None:
            self.state_history = []
        if self.entities is None:
            self.entities = {}


class AgenticStateMachine:
    """
    Multi-hop reasoning state machine for RAG.

    Workflow:
    1. INITIAL: Check if initial results meet quality threshold
    2. STEP_BACK: Generate broader mechanistic queries
    3. MECHANISTIC_SEARCH: Search with mechanistic queries
    4. VERIFY: Check if quality improved
    5. DONE: Return final results

    Triggers when: max(scores) < AGENTIC_SCORE_THRESHOLD
    """

    def __init__(
        self,
        query_generator: Any,  # LLM query generator
        hybrid_retriever: Any,  # Hybrid retriever
        web_search: Optional[Any] = None,
        web_crawler: Optional[Any] = None,
        reranker: Optional[Any] = None,
        score_threshold: Optional[float] = None,
        max_hops: Optional[int] = None
    ):
        """
        Initialize Agentic State Machine.

        Args:
            query_generator: Query generator for step-back queries
            hybrid_retriever: Hybrid retriever for searches
            web_search: WebSearchClient for agentic web expansion (ev1 parity)
            web_crawler: WebCrawler for agentic web expansion (ev1 parity)
            reranker: QwenReranker for scoring newly discovered evidence (ev1 parity)
            score_threshold: Minimum score to avoid agentic mode
            max_hops: Maximum reasoning hops
        """
        self.query_generator = query_generator
        self.hybrid_retriever = hybrid_retriever
        self.web_search = web_search
        self.web_crawler = web_crawler
        self.reranker = reranker
        self.score_threshold = score_threshold or settings.AGENTIC_SCORE_THRESHOLD
        self.max_hops = max_hops or settings.AGENTIC_MAX_HOPS

        logger.info(
            "AgenticStateMachine initialized",
            score_threshold=self.score_threshold,
            max_hops=self.max_hops,
            use_web_search=bool(self.web_search and self.web_crawler),
            use_reranker=bool(self.reranker),
        )

    async def run(
        self,
        query: str,
        initial_results: List[RetrievalResult],
        entities: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Execute agentic reasoning loop.

        Args:
            query: Original user query
            initial_results: Results from initial retrieval
            entities: Optional extracted entities (genes/drugs/keywords)

        Returns:
            Enhanced results after multi-hop reasoning
        """
        # Initialize context
        context = AgenticContext(
            query=query,
            results=initial_results.copy(),
            entities=entities or {}
        )

        # Start state machine
        state = AgenticState.INITIAL

        logger.info(
            "Starting agentic reasoning loop",
            query=query[:100],
            initial_result_count=len(initial_results),
            initial_top_score=initial_results[0].score if initial_results else 0.0
        )

        while state != AgenticState.DONE:
            context.state_history.append(state)
            logger.debug("State transition", current_state=state.value)

            if state == AgenticState.INITIAL:
                state = await self._handle_initial(context)

            elif state == AgenticState.STEP_BACK:
                state = await self._handle_step_back(context)

            elif state == AgenticState.MECHANISTIC_SEARCH:
                state = await self._handle_mechanistic_search(context)

            elif state == AgenticState.VERIFY:
                state = await self._handle_verify(context)

        logger.info(
            "Agentic reasoning completed",
            total_hops=context.hop,
            final_result_count=len(context.results),
            state_history=[s.value for s in context.state_history]
        )

        return context.results

    async def _handle_initial(self, context: AgenticContext) -> AgenticState:
        """
        Initial state: Check quality of initial results.

        Returns:
            STEP_BACK if quality below threshold, DONE otherwise
        """
        quality_score = self._check_quality(context.results)

        logger.debug(
            "Initial quality check",
            quality_score=quality_score,
            threshold=self.score_threshold
        )

        if quality_score < self.score_threshold:
            logger.info("Quality below threshold, triggering agentic mode")
            return AgenticState.STEP_BACK
        else:
            logger.info("Quality sufficient, skipping agentic mode")
            return AgenticState.DONE

    async def _handle_step_back(self, context: AgenticContext) -> AgenticState:
        """
        Step-back state: Generate broader mechanistic queries.

        Strategy:
        1. What pathway does this gene belong to?
        2. What are known paralogs/redundant genes?
        3. What stress responses occur upon gene loss?

        Returns:
            MECHANISTIC_SEARCH
        """
        logger.info("Generating step-back mechanistic queries")

        # Generate mechanistic queries via LLM
        mechanistic_queries = await self.query_generator.generate_step_back(
            context.query,
            num_queries=settings.AGENTIC_MECHANISTIC_QUERIES
        )

        context.mechanistic_queries = mechanistic_queries

        logger.info(
            "Step-back queries generated",
            num_queries=len(mechanistic_queries),
            queries=mechanistic_queries
        )

        return AgenticState.MECHANISTIC_SEARCH

    async def _handle_mechanistic_search(self, context: AgenticContext) -> AgenticState:
        """
        Mechanistic search state: Execute broader searches.

        Returns:
            VERIFY
        """
        if not context.mechanistic_queries:
            logger.warning("No mechanistic queries, skipping search")
            return AgenticState.DONE

        logger.info("Executing mechanistic searches", num_queries=len(context.mechanistic_queries))

        existing_content = {hash(r.content) for r in context.results}
        existing_links = {
            (r.metadata or {}).get("link") or (r.metadata or {}).get("url")
            for r in context.results
            if (r.metadata or {}).get("link") or (r.metadata or {}).get("url")
        }
        seen_links = {l for l in existing_links if l}

        # === Local (hybrid) retrieval ===
        local_new: List[RetrievalResult] = []
        if self.hybrid_retriever:
            try:
                local_new = await self.hybrid_retriever.search_multi(
                    context.mechanistic_queries,
                    top_k=20,
                    use_entity_boost=False
                )
            except Exception as e:
                logger.warning("Agentic local retrieval failed", error=str(e))

        # === Web expansion (ev1 parity): web_search + crawl_content ===
        web_new: List[RetrievalResult] = []
        if self.web_search and self.web_crawler:
            try:
                web_new = await self._search_web(context.mechanistic_queries, context.entities or {}, seen_links)
            except Exception as e:
                logger.warning("Agentic web retrieval failed", error=str(e))

        # Merge and deduplicate new results
        combined_new = []
        for r in (local_new or []) + (web_new or []):
            if hash(r.content) in existing_content:
                continue
            link = (r.metadata or {}).get("link") or (r.metadata or {}).get("url")
            if link and link in seen_links:
                continue
            if link:
                seen_links.add(link)
            existing_content.add(hash(r.content))
            combined_new.append(r)

        if not combined_new:
            logger.info("Agentic hop produced no new evidence; stopping", hop=context.hop + 1)
            return AgenticState.DONE

        # Rerank newly discovered evidence so scores are comparable to the initial reranked list.
        if self.reranker:
            combined_new = await self._rerank_new(query=context.query, new_results=combined_new)

        context.results.extend(combined_new)
        context.results.sort(key=lambda x: x.score, reverse=True)
        context.hop += 1

        logger.info(
            "Mechanistic search completed",
            added=len(combined_new),
            total=len(context.results),
            current_hop=context.hop,
            new_max=context.results[0].score if context.results else 0.0
        )

        return AgenticState.VERIFY

    async def _handle_verify(self, context: AgenticContext) -> AgenticState:
        """
        Verify state: Check if quality improved.

        Returns:
            DONE if max hops reached or quality sufficient, STEP_BACK otherwise
        """
        quality_score = self._check_quality(context.results)

        logger.debug(
            "Verification check",
            quality_score=quality_score,
            current_hop=context.hop,
            max_hops=self.max_hops
        )

        # Stop if max hops reached
        if context.hop >= self.max_hops:
            logger.info("Max hops reached, stopping agentic loop")
            return AgenticState.DONE

        # Stop if quality sufficient
        if quality_score >= self.score_threshold:
            logger.info("Quality threshold met, stopping agentic loop")
            return AgenticState.DONE

        # Continue for another hop
        logger.info("Quality still below threshold, continuing agentic loop")
        return AgenticState.STEP_BACK

    def _check_quality(self, results: List[RetrievalResult]) -> float:
        """
        Check quality of results.

        Quality metric: max score among top-3 results

        Args:
            results: List of retrieval results

        Returns:
            Quality score (0.0 to 1.0+)
        """
        if not results:
            return 0.0

        # Get top-3 scores
        top_scores = sorted([r.score for r in results[:3]], reverse=True)

        # Return max score (best result quality)
        return top_scores[0] if top_scores else 0.0

    def should_trigger(self, results: List[RetrievalResult]) -> bool:
        """
        Determine if agentic mode should be triggered.

        Args:
            results: Initial retrieval results

        Returns:
            True if quality below threshold
        """
        if not settings.USE_AGENTIC_RAG:
            return False

        quality_score = self._check_quality(results)
        return quality_score < self.score_threshold

    async def _search_web(
        self,
        queries: List[str],
        entities: Dict[str, Any],
        seen_links: set
    ) -> List[RetrievalResult]:
        """
        Agentic web search + crawl (mirrors ev1's agentic_retrieval_loop behavior).
        """
        import asyncio

        genes = list((entities or {}).get("genes", []) or [])[:3]
        drugs = list((entities or {}).get("drugs", []) or [])[:2]

        search_tasks = [
            self.web_search.search_biomedical(q, genes=genes, drugs=drugs, num_results=5)
            for q in (queries or [])[:4]
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        unique_hits: List[Dict[str, Any]] = []
        for res in search_results:
            if isinstance(res, Exception):
                logger.warning("Agentic web search task failed", error=str(res))
                continue
            for item in res:
                link = (item.get("link") or "").strip()
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                unique_hits.append(item)

        # Keep latency bounded.
        unique_hits = unique_hits[:20]
        if not unique_hits:
            return []

        semaphore = asyncio.Semaphore(5)

        async def _crawl_one(hit: Dict[str, Any]) -> Optional[RetrievalResult]:
            link = hit.get("link", "")
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")
            date = (hit.get("date") or "").strip() or "Unknown Date"

            async with semaphore:
                crawled_content = None
                if link:
                    try:
                        crawled_content = await self.web_crawler.crawl_content(link, title=title)
                    except Exception as e:
                        logger.debug("Agentic web crawl failed", url=link, error=str(e))

                if crawled_content:
                    body_content = crawled_content[:3000]
                    content_source = "Deep Fetch (Full Abstract)"
                else:
                    body_content = snippet
                    content_source = "Google Snippet"

                if not body_content:
                    return None

                content = (
                    f"Title: {title}\n"
                    f"Date: {date}\n"
                    f"Link: {link}\n"
                    f"Content Type: {content_source}\n"
                    f"Body: {body_content}\n"
                )

                source_type = SourceType.PUBMED if "PMID:" in content else SourceType.WEB_SEARCH
                return RetrievalResult(
                    content=content,
                    metadata={
                        "paper_title": title,
                        "title": title,
                        "date": date,
                        "source": "web_search",
                        "key_genes": [],
                        "link": link,
                        "url": link,
                        "snippet": snippet,
                        "is_web": True
                    },
                    score=0.0,
                    source=source_type
                )

        crawl_tasks = [_crawl_one(hit) for hit in unique_hits]
        crawled = await asyncio.gather(*crawl_tasks, return_exceptions=True)

        results: List[RetrievalResult] = []
        for item in crawled:
            if isinstance(item, Exception):
                continue
            if item:
                results.append(item)

        return results

    async def _rerank_new(self, query: str, new_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank new evidence using QwenReranker on a background thread pool."""
        import asyncio

        if not new_results or not self.reranker:
            return new_results

        semaphore = asyncio.Semaphore(4)

        async def _score_one(item: RetrievalResult) -> RetrievalResult:
            async with semaphore:
                try:
                    score = await asyncio.to_thread(
                        self.reranker.compute_score,
                        query,
                        item.content,
                        None,
                        "synthetic_lethality"
                    )
                    item.score = float(score)
                except Exception as e:
                    logger.debug("Agentic rerank failed for one item", error=str(e))
            return item

        rescored = await asyncio.gather(*[_score_one(r) for r in new_results])
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored
