"""
Context Assembler with Waterfall Token Budget Allocation

Implements P0 > P1 > P2 > P3 priority system for context allocation.
Matches the original SLAgent_backend_dspya2.py waterfall implementation.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import settings
from utils.logger import get_logger
from retrieval import RetrievalResult, SourceType

logger = get_logger(__name__)


@dataclass
class ContextSegment:
    """A segment of context with priority"""
    priority: int  # 0 = highest
    label: str
    content: str
    token_count: int
    source: SourceType


class ContextAssembler:
    """
    Dynamic context assembly with Waterfall token allocation.

    Priority Levels:
    - P0: External KG (1500 tokens max, highest confidence)
    - P1: Structured KB / DepMap (3000 tokens max)
    - P2: Graph RAG (1500 tokens max)
    - P3: Literature (remaining budget, lowest confidence)
    """

    def __init__(self, token_counter: Any):
        """
        Initialize Context Assembler.

        Args:
            token_counter: Token counting utility
        """
        self.token_counter = token_counter

        # Budget allocation (from settings)
        self.max_total_tokens = settings.MAX_CONTEXT_TOKENS
        self.system_tokens = settings.TOKEN_BUDGET_SYSTEM
        self.p0_max = settings.TOKEN_BUDGET_EXTERNAL_KG_MAX
        self.p1_max = settings.TOKEN_BUDGET_KB_MAX
        self.p2_max = settings.TOKEN_BUDGET_GRAPH_MAX

        logger.info(
            "ContextAssembler initialized",
            max_total=self.max_total_tokens,
            p0_max=self.p0_max,
            p1_max=self.p1_max,
            p2_max=self.p2_max
        )

    async def assemble(
        self,
        query: str,
        results: Dict[SourceType, List[RetrievalResult]],
        external_kg_context: str = "",
        graph_context: str = "",
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, int], List[Dict[str, Any]]]:
        """
        Assemble context from retrieval results with waterfall allocation.

        Priority Order (matching original implementation):
        - P0: External KG (highest priority, most trusted)
        - P1: Structured KB / DepMap (known SL pairs)
        - P2: Graph RAG (dynamic relationships)
        - P3: Literature (remaining budget)

        Args:
            query: Original user query
            results: Retrieval results grouped by source
            external_kg_context: Pre-formatted external KG context (optional)
            graph_context: Pre-formatted graph RAG context (optional)
            system_prompt: System instruction template (optional)

        Returns:
            Tuple of (assembled_context, token_stats, references)
        """
        budget_remaining = self.max_total_tokens - self.system_tokens

        logger.info(
            "Starting waterfall context assembly",
            initial_budget=budget_remaining,
            num_sources=len(results)
        )

        stats = {
            "p0_tokens": 0,
            "p1_tokens": 0,
            "p2_tokens": 0,
            "p3_tokens": 0,
            "total_tokens": 0
        }

        references: List[Dict[str, Any]] = []
        context_parts = []

        # Add date header (for judging 'recent' evidence)
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_header = f"**Current Date:** {current_date} (Use this to judge 'recent' evidence)\n\n"
        context_parts.append(date_header)

        current_token_count = self.system_tokens

        # === P0: External Knowledge Graph (Highest Priority) ===
        if external_kg_context:
            p0_tokens = await self.token_counter.count(external_kg_context)

            # Truncate if exceeds budget
            if p0_tokens > self.p0_max:
                external_kg_context = await self._truncate_to_budget(
                    external_kg_context, self.p0_max
                )
                p0_tokens = await self.token_counter.count(external_kg_context)

            if p0_tokens > 0:
                context_parts.append("=== [PRIORITY 0] External Knowledge Graph (Highest Confidence) ===\n")
                context_parts.append(external_kg_context)
                context_parts.append("\n\n")
                stats["p0_tokens"] = p0_tokens
                current_token_count += p0_tokens
                logger.debug("P0 (External KG) allocated", tokens=p0_tokens)

        # Fallback: Get External KG from results if not provided
        elif SourceType.EXTERNAL_KG in results:
            p0_segment, p0_tokens = await self._format_external_kg(
                results[SourceType.EXTERNAL_KG],
                min(budget_remaining, self.p0_max)
            )
            if p0_segment:
                context_parts.append("=== [PRIORITY 0] External Knowledge Graph (Highest Confidence) ===\n")
                context_parts.append(p0_segment.content)
                context_parts.append("\n\n")
                stats["p0_tokens"] = p0_tokens
                current_token_count += p0_tokens
                budget_remaining -= p0_tokens
                logger.debug("P0 (External KG) allocated", tokens=p0_tokens)

        # === P1: Structured KB / DepMap (Known SL Pairs) ===
        if SourceType.DEPMAP_KB in results:
            kb_results = results[SourceType.DEPMAP_KB]

            # Separate synthetic docs (from KB) vs literature
            synthetic_docs = [r for r in kb_results if r.metadata.get('is_synthetic', False)]

            if synthetic_docs:
                p1_budget = min(budget_remaining, self.p1_max)
                p1_segment, p1_tokens = await self._format_depmap_kb(
                    synthetic_docs, p1_budget
                )

                if p1_segment:
                    context_parts.append("=== [PRIORITY 1] DepMap Knowledge Base (Known SL Pairs) ===\n")
                    context_parts.append(p1_segment.content)
                    context_parts.append("\n\n")
                    stats["p1_tokens"] = p1_tokens
                    current_token_count += p1_tokens
                    budget_remaining -= p1_tokens
                    logger.debug("P1 (DepMap KB) allocated", tokens=p1_tokens, entries=len(synthetic_docs))

        # === P2: Graph RAG (Dynamic Relationships) ===
        if graph_context:
            p2_tokens = await self.token_counter.count(graph_context)

            # Truncate if exceeds budget
            if p2_tokens > self.p2_max:
                graph_context = await self._truncate_graph_context(
                    graph_context, self.p2_max
                )
                p2_tokens = await self.token_counter.count(graph_context)

            # Check waterfall budget
            if current_token_count + p2_tokens > self.max_total_tokens - 500:
                # Budget tight, further truncate
                remaining_budget = max(500, self.max_total_tokens - current_token_count - 500)
                if remaining_budget < p2_tokens:
                    graph_context = await self._truncate_graph_context(
                        graph_context, remaining_budget
                    )
                    p2_tokens = await self.token_counter.count(graph_context)

            if p2_tokens > 0:
                context_parts.append("=== [PRIORITY 2] Dynamic Knowledge Graph ===\n")
                context_parts.append(graph_context)
                context_parts.append("\n\n")
                stats["p2_tokens"] = p2_tokens
                current_token_count += p2_tokens
                budget_remaining -= p2_tokens
                logger.debug("P2 (Graph RAG) allocated", tokens=p2_tokens)

        # Fallback: Get Graph RAG from results if not provided
        elif SourceType.GRAPH_RAG in results:
            p2_budget = min(budget_remaining, self.p2_max)
            p2_segment, p2_tokens = await self._format_graph_rag(
                results[SourceType.GRAPH_RAG],
                p2_budget
            )
            if p2_segment:
                context_parts.append("=== [PRIORITY 2] Dynamic Knowledge Graph ===\n")
                context_parts.append(p2_segment.content)
                context_parts.append("\n\n")
                stats["p2_tokens"] = p2_tokens
                current_token_count += p2_tokens
                budget_remaining -= p2_tokens
                logger.debug("P2 (Graph RAG) allocated", tokens=p2_tokens)

        # === P3: Literature (Remaining Budget) ===
        literature_token_budget = self.max_total_tokens - current_token_count

        # Combine all literature sources (exclude synthetic KB docs)
        literature_results = []
        for source_type in [SourceType.LOCAL_DENSE, SourceType.BM25, SourceType.PUBMED, SourceType.WEB_SEARCH]:
            if source_type in results:
                for r in results[source_type]:
                    if not r.metadata.get('is_synthetic', False):
                        literature_results.append(r)

        # Also include non-synthetic DepMap results
        if SourceType.DEPMAP_KB in results:
            for r in results[SourceType.DEPMAP_KB]:
                if not r.metadata.get('is_synthetic', False):
                    literature_results.append(r)

        if literature_results:
            context_parts.append("=== [PRIORITY 3] Retrieved Scientific Literature ===\n")

            # ev1 parity: preserve the post-diversification order when ranks are present.
            if any(getattr(r, "rank", None) is not None for r in literature_results):
                literature_results.sort(key=lambda x: x.rank if x.rank is not None else 10**9)
            else:
                literature_results.sort(key=lambda x: x.score, reverse=True)

            p3_tokens = 0

            for i, res in enumerate(literature_results):
                metadata = res.metadata or {}
                title = metadata.get('title', metadata.get('paper_title', 'Unknown'))
                score_info = f"[Rel: {res.score:.3f}]"

                # Determine source type
                if metadata.get('is_web') or res.source == SourceType.WEB_SEARCH:
                    source_type_str = f"🌐 [WEB] ({metadata.get('link', '')})"
                    genes_info = ""
                else:
                    source_type_str = "📄 [LOCAL]"
                    genes = ", ".join(metadata.get('key_genes', [])[:3])
                    genes_info = f"Key Genes: {genes}\n" if genes else ""

                chunk_text = (
                    f"\nSource [{i+1}] {source_type_str} {score_info}\n"
                    f"Title: {title}\n{genes_info}Content: {res.content}\n"
                )

                chunk_tokens = await self.token_counter.count(chunk_text)
                if p3_tokens + chunk_tokens > literature_token_budget:
                    logger.debug(f"Literature budget reached at source [{i+1}]")
                    break

                context_parts.append(chunk_text)
                p3_tokens += chunk_tokens

                # Build reference JSON
                content_for_json = res.content
                if settings.EVAL_MODE:
                    if settings.EVAL_CONTENT_MAX_LENGTH > 0:
                        if len(content_for_json) > settings.EVAL_CONTENT_MAX_LENGTH:
                            content_for_json = content_for_json[:settings.EVAL_CONTENT_MAX_LENGTH] + "..."
                        else:
                            content_for_json = content_for_json[:settings.EVAL_CONTENT_MAX_LENGTH]
                else:
                    content_for_json = content_for_json[:200] + "..."

                references.append({
                    "id": i + 1,
                    "title": title,
                    "score": res.score,
                    "content": content_for_json,
                    "link": metadata.get('link', '')
                })

            stats["p3_tokens"] = p3_tokens
            current_token_count += p3_tokens
            logger.debug("P3 (Literature) allocated", tokens=p3_tokens, chunks=len(references))

        # Assemble final context
        assembled_context = "".join(context_parts)
        stats["total_tokens"] = current_token_count

        logger.info(
            "Waterfall context assembly completed",
            p0_external_kg=stats["p0_tokens"],
            p1_kb=stats["p1_tokens"],
            p2_graph=stats["p2_tokens"],
            p3_literature=stats["p3_tokens"],
            total=stats["total_tokens"],
            budget=self.max_total_tokens
        )

        # NOTE: references are derived from the same budget-bounded literature chunks
        # that are included in the assembled context (ev1 parity).
        return assembled_context, stats, references

    async def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget, keeping complete lines."""
        lines = text.split('\n')
        truncated_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = await self.token_counter.count(line + '\n')
            if current_tokens + line_tokens <= max_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break

        return '\n'.join(truncated_lines)

    async def _truncate_graph_context(self, graph_context: str, max_tokens: int) -> str:
        """
        Truncate graph context while preserving complete triplet lines.

        Matching original _truncate_graph_context implementation.
        """
        if not graph_context:
            return ""

        current_tokens = await self.token_counter.count(graph_context)
        if current_tokens <= max_tokens:
            return graph_context

        # Split into header and triplet lines
        lines = graph_context.split('\n')
        header_lines = []
        triplet_lines = []

        for line in lines:
            if '--[' in line and ']-->' in line:
                triplet_lines.append(line)
            else:
                header_lines.append(line)

        # Header is always kept
        header = '\n'.join(header_lines)
        header_tokens = await self.token_counter.count(header)

        # Calculate remaining budget for triplets
        remaining_tokens = max_tokens - header_tokens - 50  # Buffer

        if remaining_tokens <= 0:
            return ""

        # Add triplets until budget exhausted
        kept_triplets = []
        current_triplet_tokens = 0

        for triplet in triplet_lines:
            triplet_tokens = await self.token_counter.count(triplet + '\n')
            if current_triplet_tokens + triplet_tokens <= remaining_tokens:
                kept_triplets.append(triplet)
                current_triplet_tokens += triplet_tokens
            else:
                break

        if not kept_triplets:
            return ""

        truncated = header + '\n' + '\n'.join(kept_triplets)
        logger.debug(
            "Graph context truncated",
            original_tokens=current_tokens,
            final_tokens=await self.token_counter.count(truncated)
        )

        return truncated

    async def _format_external_kg(
        self,
        results: List[RetrievalResult],
        max_tokens: int
    ) -> tuple[Optional[ContextSegment], int]:
        """Format External KG results (P0 priority)"""
        if not results:
            return None, 0

        formatted_lines = ["**External Knowledge Graph (Highest Confidence):**\n"]

        used_tokens = 0
        for i, result in enumerate(results, 1):
            # Format: "1. Gene A → Gene B (relation: CelllinedependOnGene, score: 0.95)"
            relation = result.metadata.get('relation', 'unknown')
            score = result.score

            line = f"{i}. {result.content} (relation: {relation}, score: {score:.2f})\n"
            line_tokens = await self.token_counter.count(line)

            if used_tokens + line_tokens > max_tokens:
                formatted_lines.append(f"\n...(truncated, {len(results) - i + 1} more entries)")
                break

            formatted_lines.append(line)
            used_tokens += line_tokens

        content = "".join(formatted_lines)

        segment = ContextSegment(
            priority=0,
            label="External Knowledge Graph",
            content=content,
            token_count=used_tokens,
            source=SourceType.EXTERNAL_KG
        )

        return segment, used_tokens

    async def _format_depmap_kb(
        self,
        results: List[RetrievalResult],
        max_tokens: int
    ) -> tuple[Optional[ContextSegment], int]:
        """Format DepMap KB results (P1 priority)"""
        if not results:
            return None, 0

        formatted_lines = ["**DepMap Knowledge Base:**\n"]

        used_tokens = 0
        for i, result in enumerate(results, 1):
            line = f"{i}. {result.content}\n"
            line_tokens = await self.token_counter.count(line)

            if used_tokens + line_tokens > max_tokens:
                formatted_lines.append(f"\n...(truncated)")
                break

            formatted_lines.append(line)
            used_tokens += line_tokens

        content = "".join(formatted_lines)

        segment = ContextSegment(
            priority=1,
            label="DepMap Knowledge Base",
            content=content,
            token_count=used_tokens,
            source=SourceType.DEPMAP_KB
        )

        return segment, used_tokens

    async def _format_graph_rag(
        self,
        results: List[RetrievalResult],
        max_tokens: int
    ) -> tuple[Optional[ContextSegment], int]:
        """Format Graph RAG triplets (P2 priority)"""
        if not results:
            return None, 0

        formatted_lines = ["**Extracted Knowledge Graph:**\n"]

        used_tokens = 0
        for i, result in enumerate(results, 1):
            # Triplet format: (entity1, relation, entity2)
            line = f"{i}. {result.content}\n"
            line_tokens = await self.token_counter.count(line)

            if used_tokens + line_tokens > max_tokens:
                formatted_lines.append(f"\n...(truncated)")
                break

            formatted_lines.append(line)
            used_tokens += line_tokens

        content = "".join(formatted_lines)

        segment = ContextSegment(
            priority=2,
            label="Dynamic Knowledge Graph",
            content=content,
            token_count=used_tokens,
            source=SourceType.GRAPH_RAG
        )

        return segment, used_tokens

    async def _format_literature(
        self,
        results: List[RetrievalResult],
        max_tokens: int
    ) -> tuple[Optional[ContextSegment], int]:
        """Format literature results (P3 priority)"""
        if not results:
            return None, 0

        formatted_lines = ["**Literature Evidence:**\n"]

        used_tokens = 0
        for i, result in enumerate(results, 1):
            # Format: "[1] Title: ... | Authors: ... | Abstract: ..."
            metadata = result.metadata
            title = metadata.get('title', 'N/A')
            authors = metadata.get('authors', 'N/A')
            pmid = metadata.get('pmid', '')
            source = metadata.get('source', result.source.value)

            line = f"[{i}] **Title:** {title}\n"
            line += f"   **Authors:** {authors}\n"
            if pmid:
                line += f"   **PMID:** {pmid}\n"
            line += f"   **Source:** {source}\n"
            line += f"   **Content:** {result.content[:500]}...\n\n"  # Truncate content

            line_tokens = await self.token_counter.count(line)

            if used_tokens + line_tokens > max_tokens:
                formatted_lines.append(f"\n...(truncated, {len(results) - i + 1} more references)")
                break

            formatted_lines.append(line)
            used_tokens += line_tokens

        content = "".join(formatted_lines)

        segment = ContextSegment(
            priority=3,
            label="Literature References",
            content=content,
            token_count=used_tokens,
            source=SourceType.LOCAL_DENSE
        )

        return segment, used_tokens
