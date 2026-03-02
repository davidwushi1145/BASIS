"""
Query Generator - Generate Multi-Aspect Search Queries

Generates 4 diverse queries for comprehensive literature search.
"""

import re
from typing import Any, List, Optional, Dict
from llm.manager import LLMManager
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class QueryGenerator:
    """
    Generate optimized search queries for biomedical literature.

    Features:
    - 4-aspect search strategy
    - Gene synonym expansion integration
    - Fallback to original query
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        entity_analyzer: Any = None
    ):
        """
        Initialize Query Generator.

        Args:
            llm_manager: LLM manager instance
            entity_analyzer: Entity analyzer for gene expansion
        """
        self.llm = llm_manager
        self.entity_analyzer = entity_analyzer

        logger.info("QueryGenerator initialized")

    async def generate(
        self,
        query: str,
        entities: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate multi-aspect search queries.

        Args:
            query: Original user query

        Returns:
            List of 4 optimized search queries
        """
        logger.info("Generating search queries", query=query[:100])

        queries = []

        # Step 1: Extract genes and generate synonym query (if enabled)
        if self.entity_analyzer and settings.USE_GENE_SYNONYM_EXPANSION:
            effective_entities = entities
            if effective_entities is None:
                effective_entities = await self.entity_analyzer.extract(query)

            genes = list(effective_entities.get('genes', []))

            if genes:
                gene_alias_map = await self.entity_analyzer.expand_gene_aliases(genes)
                if gene_alias_map:
                    expanded_query = self.entity_analyzer.build_expanded_query(
                        query,
                        gene_alias_map
                    )
                    queries.append(expanded_query)
                    logger.debug("Added synonym-expanded query", length=len(expanded_query))

        # Step 2: Generate LLM-based queries
        try:
            response = await self.llm.generate_with_template(
                "query_generation",
                {"question": query},
                temperature=0.4,
                max_tokens=5000
            )

            # Parse queries (one per line)
            llm_queries = self._parse_queries(response)
            queries.extend(llm_queries)

        except Exception as e:
            logger.error("Query generation failed", error=str(e))

        # Step 3: Ensure we have at least the original query
        if not queries:
            queries = [query]

        # Limit to 4 queries
        final_queries = queries[:4]

        logger.info(
            "Queries generated",
            num_queries=len(final_queries),
            queries=[q[:80] + "..." if len(q) > 80 else q for q in final_queries]
        )

        return final_queries

    async def generate_step_back(
        self,
        query: str,
        num_queries: int = 3
    ) -> List[str]:
        """
        Generate step-back mechanistic queries for agentic RAG.

        Args:
            query: Original query
            num_queries: Number of queries to generate

        Returns:
            List of broader mechanistic queries
        """
        logger.info("Generating step-back queries", query=query[:100])

        try:
            response = await self.llm.generate_with_template(
                "step_back",
                {"query": query, "num_queries": num_queries},
                temperature=0.5,  # Higher temperature for diversity
                max_tokens=3000
            )

            queries = self._parse_queries(response)

            # Fallback: generate generic mechanistic queries
            if not queries:
                queries = self._fallback_step_back(query)

            logger.info(
                "Step-back queries generated",
                num_queries=len(queries),
                queries=queries
            )

            return queries[:num_queries]

        except Exception as e:
            logger.error("Step-back generation failed", error=str(e))
            return self._fallback_step_back(query)

    def _parse_queries(self, response: str) -> List[str]:
        """
        Parse queries from LLM response.

        Args:
            response: LLM output text

        Returns:
            List of query strings
        """
        queries = []

        for line in response.split('\n'):
            # Remove numbering, bullets, etc.
            clean = re.sub(r'^[\d\.\-\*\s]+', '', line.strip())

            # Skip empty lines and headers
            if clean and not clean.startswith(('#', 'Example', 'Output')):
                queries.append(clean)

        return queries

    def _fallback_step_back(self, query: str) -> List[str]:
        """
        Fallback step-back queries when LLM fails.

        Args:
            query: Original query

        Returns:
            Generic mechanistic queries
        """
        # Extract first gene name if possible
        entities = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', query)
        gene = entities[0] if entities else "gene"

        queries = [
            f"{gene} pathway function mechanism",
            f"{gene} paralog redundancy backup",
            f"{gene} loss cellular stress phenotype"
        ]

        logger.warning("Using fallback step-back queries", queries=queries)

        return queries
