"""
Intent Detector - Classify User Intent

Determines if query is GENERAL (casual chat) or SCIENTIFIC (biomedical question).
"""

from typing import Literal
from llm.manager import LLMManager
from utils.logger import get_logger
from utils.cache import LRUCache

logger = get_logger(__name__)


class IntentDetector:
    """
    Classify user queries into GENERAL or SCIENTIFIC.

    Caches recent results to avoid redundant API calls.
    """

    def __init__(self, llm_manager: LLMManager, cache_size: int = 100):
        """
        Initialize Intent Detector.

        Args:
            llm_manager: LLM manager instance
            cache_size: LRU cache size for results
        """
        self.llm = llm_manager
        self.cache = LRUCache(cache_size)

        logger.info("IntentDetector initialized", cache_size=cache_size)

    async def detect(self, query: str) -> Literal["GENERAL", "SCIENTIFIC"]:
        """
        Detect intent of user query.

        Args:
            query: User query string

        Returns:
            "GENERAL" or "SCIENTIFIC"
        """
        # Check cache
        cache_key = query.strip().lower()
        if cache_key in self.cache:
            cached_intent = self.cache[cache_key]
            logger.debug("Intent cache hit", query=query[:50], intent=cached_intent)
            return cached_intent

        logger.debug("Detecting intent", query=query[:100])

        # Generate using template
        response = await self.llm.generate_with_template(
            "intent_detection",
            {"query": query},
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=100
        )

        # Parse response
        intent = response.strip().upper()

        # Fallback to SCIENTIFIC if unclear
        if "SCIENTIFIC" in intent:
            final_intent = "SCIENTIFIC"
        elif "GENERAL" in intent:
            final_intent = "GENERAL"
        else:
            logger.warning(
                "Unclear intent response, defaulting to SCIENTIFIC",
                response=response
            )
            final_intent = "SCIENTIFIC"

        # Cache result
        self.cache[cache_key] = final_intent

        logger.info("Intent detected", query=query[:50], intent=final_intent)

        return final_intent
