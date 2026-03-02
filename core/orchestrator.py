"""
Main Orchestrator - Task Coordination Engine

Coordinates the full RAG pipeline from intent detection to response generation.
"""

import asyncio
from typing import AsyncGenerator, Dict, List, Any, Optional
from uuid import uuid4

from config import settings
from retrieval import RetrievalResult, SourceType
from core.context_assembler import ContextAssembler
from core.agentic_state_machine import AgenticStateMachine
from utils.logger import get_logger, set_request_id

logger = get_logger(__name__)


class Orchestrator:
    """
    Main task orchestrator for SLAgent.

    Pipeline:
    1. Intent detection
    2. Entity extraction
    3. Query generation
    4. Local + web candidate retrieval
    5. Reranking and optional agentic loop
    6. DepMap injection + entity boost + diversity filter
    7. Context compression (optional)
    8. External KG + Graph RAG context build
    9. Context assembly (waterfall budgets)
    10. Solver + optional verifier loop
    """

    def __init__(
        self,
        hybrid_retriever: Optional[Any],
        entity_analyzer: Optional[Any],
        intent_detector: Any,
        query_generator: Any,
        context_assembler: ContextAssembler,
        solver: Any,
        verifier: Optional[Any] = None,
        agentic_state_machine: Optional[AgenticStateMachine] = None,
        external_kg: Optional[Any] = None,
        depmap_kb: Optional[Any] = None,
        graph_rag: Optional[Any] = None,
        web_search: Optional[Any] = None,
        web_crawler: Optional[Any] = None,
        reranker: Optional[Any] = None,
        compressor: Optional[Any] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.entity_analyzer = entity_analyzer
        self.intent_detector = intent_detector
        self.query_generator = query_generator
        self.context_assembler = context_assembler
        self.solver = solver
        self.verifier = verifier
        self.agentic_sm = agentic_state_machine
        self.external_kg = external_kg
        self.depmap_kb = depmap_kb
        self.graph_rag = graph_rag
        self.web_search = web_search
        self.web_crawler = web_crawler
        self.reranker = reranker
        self.compressor = compressor

        logger.info(
            "Orchestrator initialized",
            use_hybrid=hybrid_retriever is not None,
            use_web_search=web_search is not None,
            use_reranker=reranker is not None,
            use_verifier=verifier is not None,
            use_external_kg=external_kg is not None,
            use_depmap_kb=depmap_kb is not None,
            use_graph_rag=graph_rag is not None,
            use_compressor=compressor is not None
        )

    async def process_query(
        self,
        query: str,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query and stream NDJSON-style events."""
        if not request_id:
            request_id = str(uuid4())
        set_request_id(request_id)
        logger.info("Processing query", query=query[:100], request_id=request_id)

        try:
            yield {"type": "progress", "data": "Detecting intent..."}
            intent = await self.intent_detector.detect(query)
            logger.info("Intent detected", intent=intent)

            if intent == "GENERAL":
                yield {"type": "references", "data": []}
                async for event in self._handle_general_chat(query):
                    yield event
                return

            yield {"type": "progress", "data": "Extracting entities..."}
            entities = await self._extract_entities(query)
            genes = entities.get("genes", [])
            drugs = entities.get("drugs", [])

            yield {"type": "progress", "data": "Generating search queries..."}
            sub_queries = await self.query_generator.generate(query, entities=entities)
            if query not in sub_queries:
                sub_queries = [query] + sub_queries
            sub_queries = sub_queries[:4]
            logger.info("Queries generated", count=len(sub_queries))

            hyde_doc = None
            if (
                settings.USE_HYDE
                and self.hybrid_retriever
                and getattr(self.hybrid_retriever, "hyde_generator", None)
            ):
                yield {"type": "progress", "data": "Generating HyDE query expansion..."}
                hyde_doc = await self._generate_hyde_document(query)

            yield {"type": "progress", "data": "Searching local and web sources..."}
            candidates = await self._collect_candidates(query, sub_queries, entities, hyde_doc=hyde_doc)
            logger.info("Candidate retrieval done", count=len(candidates))

            if not candidates:
                yield {"type": "references", "data": []}
                yield {"type": "progress", "data": "No direct evidence found, switching to general scientific response..."}
                prompt_override = (
                    f"User Query: {query}\n"
                    "No specific papers found. Answer based on general scientific knowledge."
                )
                async for event in self.solver.generate_stream(
                    query,
                    "",
                    prompt_override=prompt_override,
                    temperature=0.7
                ):
                    yield event
                return

            yield {"type": "progress", "data": "Reranking candidates..."}
            scored = await self._rerank_candidates(query, candidates)

            if self.agentic_sm and self.agentic_sm.should_trigger(scored):
                yield {"type": "progress", "data": "Triggering multi-hop reasoning..."}
                scored = await self.agentic_sm.run(query, scored, entities=entities)
                scored.sort(key=lambda x: x.score, reverse=True)

            # DepMap synthetic docs are injected after rerank, matching the original sequence.
            scored = await self._inject_depmap_synthetic(scored, genes)
            scored = await self._apply_entity_boost(query, scored, entities)
            final_results = self._diversify_results(scored)

            fallback_to_general = (not final_results) or (final_results[0].score < settings.SCORE_THRESHOLD)
            fallback_prompt_override: Optional[str] = None
            references: List[Dict[str, Any]] = []

            if fallback_to_general:
                yield {"type": "references", "data": []}
                yield {"type": "progress", "data": "Retrieved evidence below relevance threshold, switching to general scientific response..."}
                context = "No Context."
                fallback_prompt_override = (
                    f"Context:\n{context}\n\n"
                    f"User Query: {query}\n"
                    "No specific papers found in database. Answer based on general scientific knowledge."
                )
            else:
                if settings.USE_CONTEXT_COMPRESSION and self.compressor:
                    yield {"type": "progress", "data": "Compressing retrieved context..."}
                    final_results = await self._compress_results(query, final_results, genes)

                yield {"type": "progress", "data": "Building external KG context..."}
                external_kg_context = await self._build_external_kg_context(genes, drugs)

                yield {"type": "progress", "data": "Building graph context..."}
                graph_context = await self._build_graph_context(final_results, entities)

                grouped_results = self._group_by_source(final_results)

                yield {"type": "progress", "data": "Assembling context..."}
                context, token_stats, references = await self.context_assembler.assemble(
                    query,
                    grouped_results,
                    external_kg_context=external_kg_context,
                    graph_context=graph_context
                )
                logger.info("Context assembled", **token_stats)

                yield {"type": "references", "data": references}

            yield {"type": "progress", "data": "Generating draft answer..."}
            solver_response = await self._generate_answer_silently(
                query,
                context,
                prompt_override=fallback_prompt_override
            )
            final_answer = solver_response

            if self.verifier:
                max_verify_rounds = 5
                for verify_round in range(max_verify_rounds):
                    yield {"type": "progress", "data": f"Verifying answer quality (round {verify_round + 1})..."}
                    verification = await self.verifier.verify(query, context, solver_response)

                    raw_critique = verification.get("raw_response")
                    if raw_critique:
                        yield {"type": "thinking", "data": f"[Critique Round {verify_round + 1}]\n{raw_critique}\n"}

                    if verification["status"] == "PASS":
                        logger.info("Verification passed", round=verify_round + 1)
                        final_answer = solver_response
                        break

                    final_answer = solver_response
                    if not self.verifier.should_regenerate(verification):
                        logger.info("Verification low confidence, skipping regeneration")
                        break

                    if verify_round < max_verify_rounds - 1:
                        regen_prompt = await self.verifier.verify_with_regeneration_prompt(
                            query, context, solver_response, verification
                        )
                        if regen_prompt:
                            yield {"type": "progress", "data": "Refining answer based on verification feedback..."}
                            solver_response = await self._generate_answer_silently(
                                query,
                                context,
                                feedback=[regen_prompt]
                            )
                            final_answer = solver_response

            yield {"type": "progress", "data": "Generating final report..."}
            async for event in self._stream_text_tokens(final_answer):
                yield event

            logger.info("Query processing completed", request_id=request_id)

        except Exception as e:
            logger.error("Query processing failed", error=str(e), exc_info=True)
            yield {"type": "error", "data": f"Internal error: {str(e)}"}

    async def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract and normalize entity lists for downstream use."""
        if not self.entity_analyzer:
            return {"genes": [], "drugs": [], "keywords": []}

        raw = await self.entity_analyzer.extract(query)

        # Normalize to deterministic lists for slicing/serialization.
        genes = sorted(list(raw.get("genes", [])))
        drugs = sorted(list(raw.get("drugs", [])))
        keywords = sorted(list(raw.get("keywords", [])))
        diseases = sorted(list(raw.get("diseases", [])))

        return {
            "genes": genes,
            "drugs": drugs,
            "keywords": keywords,
            "diseases": diseases
        }

    async def _collect_candidates(
        self,
        query: str,
        sub_queries: List[str],
        entities: Dict[str, List[str]],
        hyde_doc: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Collect candidate results from local retriever and web retrieval."""
        tasks = [
            self._query_hybrid([query], hyde_doc=hyde_doc),
            self._query_web_sources(sub_queries, entities)
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        merged: List[RetrievalResult] = []
        seen = set()
        for result in results_list:
            if isinstance(result, Exception):
                logger.warning("Candidate retrieval task failed", error=str(result))
                continue

            for item in result:
                key = hash(item.content)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

    async def _query_hybrid(
        self,
        sub_queries: List[str],
        hyde_doc: Optional[str] = None
    ) -> List[RetrievalResult]:
        if not self.hybrid_retriever:
            return []
        try:
            kwargs: Dict[str, Any] = {}
            if hyde_doc:
                kwargs["hyde_doc"] = hyde_doc
            # IMPORTANT (ev1 parity): entity boost happens *after* reranking (and after DepMap injection),
            # not inside hybrid retrieval.
            results = await self.hybrid_retriever.search_multi(
                sub_queries,
                top_k=40,
                use_entity_boost=False,
                **kwargs
            )
            return results
        except Exception as e:
            logger.error("Hybrid retrieval failed", error=str(e))
            return []

    async def _query_web_sources(
        self,
        sub_queries: List[str],
        entities: Dict[str, List[str]]
    ) -> List[RetrievalResult]:
        if not self.web_search:
            return []

        genes = entities.get("genes", [])[:3]
        drugs = entities.get("drugs", [])[:2]

        search_tasks = [
            self.web_search.search_biomedical(q, genes=genes, drugs=drugs, num_results=5)
            for q in sub_queries[:4]
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        unique_hits = []
        seen_links = set()
        for res in search_results:
            if isinstance(res, Exception):
                logger.warning("Web search task failed", error=str(res))
                continue
            for item in res:
                link = (item.get("link") or "").strip()
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                unique_hits.append(item)

        # Limit crawling pressure to keep latency bounded.
        unique_hits = unique_hits[:20]

        semaphore = asyncio.Semaphore(5)

        async def _crawl_one(hit: Dict[str, Any]) -> Optional[RetrievalResult]:
            link = hit.get("link", "")
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")
            date = (hit.get("date") or "").strip() or "Unknown Date"

            async with semaphore:
                crawled_content = None
                if self.web_crawler and link:
                    try:
                        crawled_content = await self.web_crawler.crawl_content(link, title=title)
                    except Exception as e:
                        logger.warning("Web crawl failed", url=link, error=str(e))

                if crawled_content:
                    body_content = crawled_content[:3000]
                    content_source = "Deep Fetch (Full Abstract)"
                else:
                    body_content = snippet
                    content_source = "Google Snippet"

                if not body_content:
                    return None

                # Format content for the reranker/LLM (ev1-style web chunk).
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
                logger.warning("Web crawl task raised exception", error=str(item))
                continue
            if item:
                results.append(item)

        return results

    async def _rerank_candidates(
        self,
        query: str,
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        if not candidates:
            return []

        if not self.reranker:
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates

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
                    logger.warning("Rerank failed for one candidate", error=str(e))
                return item

        rescored = await asyncio.gather(*[_score_one(c) for c in candidates])
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored

    async def _inject_depmap_synthetic(
        self,
        results: List[RetrievalResult],
        genes: List[str]
    ) -> List[RetrievalResult]:
        if not self.depmap_kb or not settings.USE_DEPMAP_KB or not genes:
            return results

        synthetic_docs = self.depmap_kb.augment_with_depmap(genes)
        for doc in synthetic_docs:
            results.append(
                RetrievalResult(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=0.95,
                    source=SourceType.DEPMAP_KB
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def _apply_entity_boost(
        self,
        query: str,
        results: List[RetrievalResult],
        entities: Dict[str, List[str]]
    ) -> List[RetrievalResult]:
        if not settings.USE_ENTITY_BOOST or not results:
            return results

        if self.hybrid_retriever and hasattr(self.hybrid_retriever, "_apply_entity_boost"):
            try:
                boosted = await self.hybrid_retriever._apply_entity_boost(query, results, entities)
                boosted.sort(key=lambda x: x.score, reverse=True)
                return boosted
            except Exception as e:
                logger.warning("Entity boosting failed", error=str(e))

        return results

    def _diversify_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return []

        if settings.USE_ROUND_ROBIN and self.hybrid_retriever and hasattr(self.hybrid_retriever, "diversified_retrieval"):
            try:
                selected = self.hybrid_retriever.diversified_retrieval(
                    results,
                    min_papers=settings.MIN_PAPERS,
                    max_per_paper=settings.MAX_CHUNKS_PER_PAPER,
                    score_threshold=settings.SCORE_THRESHOLD
                )
                for rank, result in enumerate(selected, start=1):
                    result.rank = rank
                return selected
            except Exception as e:
                logger.warning("Diversified retrieval failed", error=str(e))

        # ev1 parity: when round-robin is disabled, fall back to simple top-k (Top-8).
        selected = results[:8] if not settings.USE_ROUND_ROBIN else results[:15]
        for rank, result in enumerate(selected, start=1):
            result.rank = rank
        return selected

    async def _compress_results(
        self,
        query: str,
        results: List[RetrievalResult],
        genes: List[str]
    ) -> List[RetrievalResult]:
        if not results:
            return results

        if hasattr(self.compressor, "compress_chunks"):
            try:
                payload = [{"content": r.content, "metadata": r.metadata} for r in results]
                compressed_chunks = await self.compressor.compress_chunks(query, payload)
                if len(compressed_chunks) == len(results):
                    for result, chunk in zip(results, compressed_chunks):
                        new_content = (chunk or {}).get("content")
                        if new_content:
                            result.content = new_content
                    return results
            except Exception as e:
                logger.warning("Batch compression failed, falling back to per-chunk", error=str(e))

        compressed: List[RetrievalResult] = []
        for result in results:
            try:
                target_tokens = 500
                new_content = await self.compressor.compress_async(
                    result.content,
                    target_tokens=target_tokens,
                    preserve_entities=genes,
                    query=query
                )
                result.content = new_content
            except Exception as e:
                logger.warning("Chunk compression failed", error=str(e))
            compressed.append(result)
        return compressed

    async def _build_external_kg_context(self, genes: List[str], drugs: List[str]) -> str:
        if not self.external_kg or not settings.USE_EXTERNAL_KG:
            return ""
        if hasattr(self.external_kg, "is_available") and not self.external_kg.is_available():
            return ""

        entities = list(genes[:5]) + list(drugs[:2])
        if not entities:
            return ""

        try:
            raw_context = self.external_kg.query_subgraph(
                entities=entities,
                hops=settings.EXTERNAL_KG_HOPS
            )
            if not raw_context:
                return ""

            max_tokens = settings.TOKEN_BUDGET_EXTERNAL_KG_MAX
            token_count = await self.context_assembler.token_counter.count(raw_context)
            if token_count <= max_tokens:
                return raw_context

            return await self.context_assembler._truncate_to_budget(raw_context, max_tokens)
        except Exception as e:
            logger.warning("External KG context build failed", error=str(e))
            return ""

    async def _build_graph_context(
        self,
        final_results: List[RetrievalResult],
        entities: Dict[str, List[str]]
    ) -> str:
        if not self.graph_rag or not settings.USE_GRAPH_RAG or not final_results:
            return ""
        if hasattr(self.graph_rag, "is_available") and not self.graph_rag.is_available():
            return ""

        if not hasattr(self.graph_rag, "build_and_validate_graph"):
            return ""

        chunks = [
            {"content": r.content, "metadata": r.metadata}
            for r in final_results[:settings.MAX_TRIPLET_CHUNKS]
        ]
        query_entities = {
            "genes": set(entities.get("genes", [])),
            "keywords": set(entities.get("keywords", []))
        }

        try:
            _, graph_context = await self.graph_rag.build_and_validate_graph(chunks, query_entities)
            if not graph_context:
                return ""

            max_tokens = settings.MAX_GRAPH_CONTEXT_TOKENS
            token_count = await self.context_assembler.token_counter.count(graph_context)
            if token_count <= max_tokens:
                return graph_context
            return await self.context_assembler._truncate_graph_context(graph_context, max_tokens)
        except Exception as e:
            logger.warning("Graph context build failed", error=str(e))
            return ""

    async def _handle_general_chat(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "progress", "data": "Responding to general query..."}
        async for event in self.solver.generate_general_chat(query):
            yield event

    async def _generate_hyde_document(self, query: str) -> Optional[str]:
        if not self.hybrid_retriever:
            return None
        hyde_generator = getattr(self.hybrid_retriever, "hyde_generator", None)
        if not hyde_generator:
            return None

        try:
            hyde_doc = await hyde_generator.generate(query)
            return hyde_doc or None
        except Exception as e:
            logger.warning("HyDE pre-generation failed", error=str(e))
            return None

    async def _generate_answer_silently(
        self,
        query: str,
        context: str,
        feedback: Optional[List[str]] = None,
        prompt_override: Optional[str] = None
    ) -> str:
        tokens: List[str] = []
        async for event in self.solver.generate_stream(
            query,
            context,
            feedback=feedback,
            prompt_override=prompt_override
        ):
            if event.get("type") == "token":
                tokens.append(event["data"])
        return "".join(tokens)

    async def _stream_text_tokens(
        self,
        text: str,
        chunk_size: int = 15
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not text:
            return

        safe_chunk_size = max(1, chunk_size)
        for i in range(0, len(text), safe_chunk_size):
            yield {"type": "token", "data": text[i:i + safe_chunk_size]}
            await asyncio.sleep(0.005)

    def _group_by_source(self, results: List[RetrievalResult]) -> Dict[SourceType, List[RetrievalResult]]:
        grouped: Dict[SourceType, List[RetrievalResult]] = {}
        for result in results:
            grouped.setdefault(result.source, []).append(result)
        return grouped

    def _format_references(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []

        for i, result in enumerate(results, 1):
            metadata = result.metadata or {}

            if settings.EVAL_MODE:
                content_for_json = result.content
                if settings.EVAL_CONTENT_MAX_LENGTH > 0 and len(content_for_json) > settings.EVAL_CONTENT_MAX_LENGTH:
                    content_for_json = content_for_json[:settings.EVAL_CONTENT_MAX_LENGTH] + "..."
            else:
                content_for_json = result.content[:200] + "..."

            title = metadata.get("title", metadata.get("paper_title", ""))
            link = metadata.get("link", metadata.get("url", ""))

            ref = {
                "id": i,
                "title": title,
                "score": round(result.score, 3),
                "content": content_for_json,
                "link": link
            }

            references.append(ref)

        return references

    async def health_check(self) -> Dict[str, Any]:
        status = {"status": "healthy", "components": {}}

        if self.hybrid_retriever:
            try:
                await self.hybrid_retriever.search("test", top_k=1)
                status["components"]["hybrid_retriever"] = "ok"
            except Exception as e:
                status["components"]["hybrid_retriever"] = f"error: {str(e)}"
                status["status"] = "unhealthy"
        else:
            status["components"]["hybrid_retriever"] = "disabled"

        status["components"]["external_kg"] = "ok" if self.external_kg else "disabled"
        status["components"]["depmap_kb"] = "ok" if self.depmap_kb else "disabled"
        status["components"]["graph_rag"] = "ok" if self.graph_rag else "disabled"
        status["components"]["web_search"] = "ok" if self.web_search else "disabled"
        status["components"]["solver"] = "ok"

        return status
