"""
Web Search Client

Google Serper API client for web search.
"""

from typing import Optional, Dict, Any, List
import re
import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchClient:
    """
    Web search client using Google Serper API.

    Features:
    - General web search
    - Scholar search
    - News search
    - Biomedical-focused search helpers
    """

    BASE_URL = "https://google.serper.dev"
    RECENT_QUERY_PATTERN = re.compile(
        r"\b(latest|recent|newest|current|up[-\s]?to[-\s]?date|202[0-9])\b|最新|近期|最近",
        re.IGNORECASE
    )

    def __init__(
        self,
        api_key: str = None,
        timeout: float = 30.0
    ):
        """
        Initialize web search client.

        Args:
            api_key: Serper API key
            timeout: Request timeout
        """
        self.api_key = api_key or getattr(settings, 'SERPER_API_KEY', None)
        self.timeout = timeout

        if not self.api_key:
            logger.warning("Serper API key not configured, web search disabled")
        else:
            logger.info("Web search client initialized")

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

    def _should_apply_recent_filter(self, query: str) -> bool:
        """Enable Serper `tbs=qdr:y` for time-sensitive queries (ev1 parity)."""
        if not query:
            return False
        return bool(self.RECENT_QUERY_PATTERN.search(query))

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search"
    ) -> List[Dict[str, Any]]:
        """
        Perform web search.

        Args:
            query: Search query
            num_results: Number of results
            search_type: "search", "scholar", or "news"

        Returns:
            List of search results
        """
        if not self.api_key:
            logger.warning("Web search unavailable - no API key")
            return []

        endpoint = f"{self.BASE_URL}/{search_type}"

        payload = {
            'q': query,
            'num': num_results
        }
        if search_type in {"search", "scholar"} and self._should_apply_recent_filter(query):
            payload["tbs"] = "qdr:y"
            logger.debug("Applying Serper date filter", query=query[:120], tbs="qdr:y")

        logger.debug("Performing web search", query=query, type=search_type, num=num_results)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    endpoint,
                    headers=self._get_headers(),
                    json=payload
                )

                if response.status_code != 200:
                    logger.warning("Web search failed", status=response.status_code)
                    return []

                data = response.json()

                # Extract organic results
                results = []
                organic = data.get('organic', [])

                for item in organic:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'date': item.get('date', ''),
                        'position': item.get('position', 0)
                    })

                logger.debug("Web search completed", query=query, results=len(results))
                return results

            except Exception as e:
                logger.error("Web search error", error=str(e))
                return []

    async def search_scholar(
        self,
        query: str,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search Google Scholar.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of scholarly articles
        """
        return await self.search(query, num_results, search_type="scholar")

    async def search_news(
        self,
        query: str,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search news articles.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of news articles
        """
        return await self.search(query, num_results, search_type="news")

    async def search_biomedical(
        self,
        query: str,
        genes: List[str] = None,
        drugs: List[str] = None,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Biomedical-focused search with site restrictions.

        Args:
            query: Base query
            genes: Gene names to include
            drugs: Drug names to include
            num_results: Number of results

        Returns:
            List of search results from biomedical sites
        """
        # Build enhanced query
        enhanced_parts = [query]

        if genes:
            gene_str = ' OR '.join(f'"{g}"' for g in genes[:3])  # Limit to 3
            enhanced_parts.append(f"({gene_str})")

        if drugs:
            drug_str = ' OR '.join(f'"{d}"' for d in drugs[:3])
            enhanced_parts.append(f"({drug_str})")

        # Restrict to biomedical sites
        site_restriction = (
            "site:ncbi.nlm.nih.gov OR "
            "site:pubmed.ncbi.nlm.nih.gov OR "
            "site:nature.com OR "
            "site:cell.com OR "
            "site:sciencedirect.com OR "
            "site:biorxiv.org OR "
            "site:medrxiv.org"
        )

        enhanced_query = f"{' '.join(enhanced_parts)} ({site_restriction})"

        return await self.search(enhanced_query, num_results)

    async def search_synthetic_lethality(
        self,
        gene1: str,
        gene2: str = None,
        context: str = None,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for synthetic lethality information.

        Args:
            gene1: Primary gene
            gene2: Secondary gene (optional)
            context: Additional context (e.g., cancer type)
            num_results: Number of results

        Returns:
            List of search results
        """
        # Build SL-focused query
        query_parts = [f'"{gene1}"']

        if gene2:
            query_parts.append(f'"{gene2}"')

        query_parts.append('"synthetic lethality" OR "synthetic lethal"')

        if context:
            query_parts.append(f'"{context}"')

        query = ' '.join(query_parts)

        return await self.search_biomedical(query, num_results=num_results)

    async def search_drug_gene(
        self,
        drug: str,
        gene: str,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for drug-gene interactions.

        Args:
            drug: Drug name
            gene: Gene name
            num_results: Number of results

        Returns:
            List of search results
        """
        query = (
            f'"{drug}" "{gene}" '
            '(inhibitor OR resistance OR sensitivity OR treatment)'
        )

        return await self.search_biomedical(query, num_results=num_results)

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as text.

        Args:
            results: List of search results

        Returns:
            Formatted string
        """
        if not results:
            return "No search results found."

        lines = ["Web Search Results:", ""]

        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.get('title', 'No title')}")
            lines.append(f"   URL: {result.get('link', 'N/A')}")
            if result.get('snippet'):
                lines.append(f"   {result['snippet']}")
            lines.append("")

        return '\n'.join(lines)

    async def get_urls_for_crawling(
        self,
        query: str,
        num_results: int = 5
    ) -> List[str]:
        """
        Get URLs from search for subsequent crawling.

        Args:
            query: Search query
            num_results: Number of URLs

        Returns:
            List of URLs
        """
        results = await self.search(query, num_results)
        return [r.get('link') for r in results if r.get('link')]

    def is_available(self) -> bool:
        """Check if web search is available."""
        return bool(self.api_key)
