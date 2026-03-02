"""
HyDE Generator - Hypothetical Document Embeddings

Generates hypothetical document answers for improved retrieval.
Based on the paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

from typing import List, Optional
import asyncio
from llm.manager import LLMManager
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ev1-aligned HyDE prompt (hypothetical abstract for dense retrieval)
HYDE_PROMPT = """
You are a computational biologist predicting potential synthetic lethal interactions.
User Query: "{query}"

Task: Write a HYPOTHETICAL scientific abstract that proposes a mechanism for this synthetic lethality based on:
1. Paralogs (e.g., ARID1A/ARID1B logic).
2. Collateral lethality (neighboring gene deletion).
3. Pathway compensation (e.g., DNA damage response).

Do NOT invent fake experimental data. Focus on the biological RATIONALE that would make a paper on this topic relevant.
Keywords to include: "compensatory pathway", "loss-of-function", "isogenic cell line", "viability assay".
"""


class HyDEGenerator:
    """
    Hypothetical Document Embeddings Generator.

    Generates hypothetical answers to improve dense retrieval.
    The generated text is embedded and used for similarity search,
    often yielding better results than raw query embeddings.
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize HyDE Generator.

        Args:
            llm_manager: LLM manager for generation
        """
        self.llm = llm_manager
        self.enabled = settings.USE_HYDE

        logger.info("HyDE Generator initialized", enabled=self.enabled)

    async def generate(self, query: str) -> str:
        """
        Compatibility wrapper used by HybridRetriever.

        Returns a single hypothetical document string.
        """
        docs = await self.generate_hypothetical_document(query, num_docs=1)
        return docs[0] if docs else ""

    async def generate_hypothetical_document(
        self,
        query: str,
        num_docs: int = 1
    ) -> List[str]:
        """
        Generate hypothetical documents for a query.

        Args:
            query: User query
            num_docs: Number of hypothetical docs to generate

        Returns:
            List of hypothetical document strings
        """
        if not self.enabled:
            logger.debug("HyDE disabled, returning empty list")
            return []

        logger.info("Generating HyDE documents", query=query[:100], num_docs=num_docs)

        hypothetical_docs = []

        for i in range(num_docs):
            try:
                prompt = HYDE_PROMPT.format(query=query)

                temperature = getattr(settings, "HYDE_TEMPERATURE", 0.9)
                max_tokens = int(getattr(settings, "HYDE_MAX_TOKENS", 25600))
                timeout_s = float(getattr(settings, "HYDE_TIMEOUT_SECONDS", 100.0))

                response = await asyncio.wait_for(
                    self.llm.generate(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ),
                    timeout=timeout_s
                )

                # Clean up response
                doc = response.strip()
                if doc:
                    hypothetical_docs.append(doc)
                    logger.debug(
                        "Generated HyDE doc",
                        index=i,
                        length=len(doc)
                    )

            except Exception as e:
                logger.warning("HyDE generation failed", index=i, error=str(e))

        logger.info("HyDE generation complete", count=len(hypothetical_docs))
        return hypothetical_docs

    async def generate_hyde_query(self, query: str) -> str:
        """
        Generate a single combined HyDE query.

        This combines the original query with a hypothetical answer
        for use in hybrid retrieval.

        Args:
            query: Original user query

        Returns:
            Enhanced query string
        """
        if not self.enabled:
            return query

        docs = await self.generate_hypothetical_document(query, num_docs=1)

        if not docs:
            return query

        # Combine original query with hypothetical doc
        combined = f"{query}\n\n{docs[0]}"

        logger.debug("Generated HyDE query", original_len=len(query), combined_len=len(combined))

        return combined

    async def generate_multi_aspect_hyde(
        self,
        query: str,
        aspects: List[str] = None
    ) -> List[str]:
        """
        Generate HyDE documents for multiple query aspects.

        Useful for complex queries that have multiple facets.

        Args:
            query: Original query
            aspects: List of aspect prompts (optional)

        Returns:
            List of hypothetical documents
        """
        if not self.enabled:
            return []

        default_aspects = [
            "mechanism",  # Biological mechanism
            "evidence",   # Experimental evidence
            "clinical",   # Clinical implications
        ]

        aspects = aspects or default_aspects

        aspect_prompts = {
            "mechanism": f"Explain the molecular mechanism: {query}",
            "evidence": f"Describe the experimental evidence for: {query}",
            "clinical": f"Discuss the clinical relevance of: {query}",
        }

        hypothetical_docs = []

        for aspect in aspects:
            aspect_query = aspect_prompts.get(aspect, f"{aspect}: {query}")

            docs = await self.generate_hypothetical_document(aspect_query, num_docs=1)
            hypothetical_docs.extend(docs)

        logger.info("Multi-aspect HyDE complete", aspects=len(aspects), docs=len(hypothetical_docs))

        return hypothetical_docs
