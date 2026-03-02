"""
bioRxiv/medRxiv Client

API client for bioRxiv and medRxiv preprint servers.
"""

import re
from typing import Optional, Dict, Any, List
import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BioRxivClient:
    """
    Client for bioRxiv/medRxiv preprint servers.

    Features:
    - DOI extraction from URLs
    - API abstract fetching
    - PubMed published version lookup
    - PDF URL generation
    """

    BASE_URL = "https://api.biorxiv.org"

    def __init__(self, timeout: float = 30.0):
        """
        Initialize bioRxiv client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        logger.info("bioRxiv client initialized")

    def extract_doi(self, url: str) -> Optional[str]:
        """
        Extract DOI from bioRxiv/medRxiv URL.

        Supports:
        - 10.1101/2020.01.14.905729 (new format)
        - 10.1101/845446 (old format)

        Args:
            url: bioRxiv/medRxiv URL

        Returns:
            DOI string or None
        """
        # Pattern for bioRxiv DOIs: 10.1101/YYYY.MM.DD.XXXXXX or 10.1101/XXXXXX
        pattern = r'(10\.1101/(?:\d{4}\.\d{2}\.\d{2}\.)?\d+)'
        match = re.search(pattern, url)

        if match:
            doi = match.group(1)
            logger.debug("Extracted DOI from URL", url=url, doi=doi)
            return doi

        return None

    async def fetch_details(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Fetch preprint details from bioRxiv API.

        Args:
            doi: bioRxiv DOI

        Returns:
            Dictionary with title, authors, abstract, date, category
        """
        # Determine server (biorxiv or medrxiv)
        server = "biorxiv"  # Default, API works for both

        url = f"{self.BASE_URL}/details/{server}/{doi}"

        logger.debug("Fetching bioRxiv details", doi=doi, url=url)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)

                if response.status_code != 200:
                    # Try medrxiv
                    url = f"{self.BASE_URL}/details/medrxiv/{doi}"
                    response = await client.get(url)

                    if response.status_code != 200:
                        logger.warning("bioRxiv API failed", doi=doi, status=response.status_code)
                        return None

                data = response.json()
                collection = data.get('collection', [])

                if not collection:
                    logger.warning("No collection in bioRxiv response", doi=doi)
                    return None

                # Get most recent version
                article = collection[-1] if collection else collection[0]

                result = {
                    'doi': article.get('doi', doi),
                    'title': article.get('title', ''),
                    'authors': article.get('authors', ''),
                    'abstract': article.get('abstract', ''),
                    'category': article.get('category', ''),
                    'date': article.get('date', ''),
                    'version': article.get('version', '1'),
                    'server': article.get('server', server)
                }

                logger.debug("bioRxiv details fetched", doi=doi, title=result['title'][:50])
                return result

            except Exception as e:
                logger.error("bioRxiv API error", doi=doi, error=str(e))
                return None

    async def fetch_abstract(self, url: str) -> Optional[str]:
        """
        Fetch abstract from bioRxiv URL.

        Args:
            url: bioRxiv/medRxiv URL

        Returns:
            Formatted abstract content
        """
        doi = self.extract_doi(url)
        if not doi:
            logger.warning("Could not extract DOI from URL", url=url)
            return None

        details = await self.fetch_details(doi)
        if not details:
            return None

        return self.format_content(details)

    def format_content(self, details: Dict[str, Any]) -> str:
        """
        Format preprint details as content string.

        Args:
            details: Details dictionary from fetch_details

        Returns:
            Formatted content string
        """
        sections = []

        sections.append(f"Title: {details.get('title', 'Unknown')}")

        if details.get('authors'):
            sections.append(f"Authors: {details['authors']}")

        if details.get('category'):
            sections.append(f"Category: {details['category']}")

        if details.get('date'):
            sections.append(f"Posted: {details['date']}")

        sections.append(f"DOI: {details.get('doi', 'N/A')}")

        if details.get('abstract'):
            sections.append(f"\nAbstract:\n{details['abstract']}")

        server = details.get('server', 'biorxiv')
        sections.append(f"\n[Source: {server}]")

        return '\n'.join(sections)

    def get_pdf_url(self, url: str) -> Optional[str]:
        """
        Generate PDF download URL from article URL.

        Args:
            url: bioRxiv/medRxiv article URL

        Returns:
            PDF URL
        """
        # Clean URL of version suffix and add .full.pdf
        clean_url = re.sub(r'v\d+$', '', url.rstrip('/'))

        # Handle content URLs
        if '/content/' in clean_url:
            if not clean_url.endswith('.pdf'):
                return clean_url + '.full.pdf'
            return clean_url

        return None

    async def search_preprints(
        self,
        query: str,
        server: str = "biorxiv",
        start_date: str = None,
        end_date: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for preprints (using content API).

        Note: bioRxiv doesn't have a proper search API,
        this fetches recent papers in a date range.

        Args:
            query: Search query (for filtering results)
            server: "biorxiv" or "medrxiv"
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum results

        Returns:
            List of matching preprint details
        """
        # Default to last 30 days if no dates specified
        if not start_date or not end_date:
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=30)
            start_date = start.strftime('%Y-%m-%d')
            end_date = end.strftime('%Y-%m-%d')

        # bioRxiv content API
        url = f"{self.BASE_URL}/details/{server}/{start_date}/{end_date}"

        logger.debug("Searching preprints", server=server, start=start_date, end=end_date)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)

                if response.status_code != 200:
                    logger.warning("bioRxiv search failed", status=response.status_code)
                    return []

                data = response.json()
                collection = data.get('collection', [])

                # Filter by query terms
                query_terms = query.lower().split()
                results = []

                for article in collection:
                    text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
                    if all(term in text for term in query_terms):
                        results.append({
                            'doi': article.get('doi', ''),
                            'title': article.get('title', ''),
                            'authors': article.get('authors', ''),
                            'abstract': article.get('abstract', ''),
                            'category': article.get('category', ''),
                            'date': article.get('date', ''),
                            'server': server
                        })

                        if len(results) >= limit:
                            break

                logger.debug("Preprint search completed", query=query, results=len(results))
                return results

            except Exception as e:
                logger.error("Preprint search error", error=str(e))
                return []

    async def check_pubmed_version(
        self,
        doi: str,
        pubmed_client=None
    ) -> Optional[str]:
        """
        Check if preprint has been published in PubMed.

        Args:
            doi: bioRxiv DOI
            pubmed_client: PubMedClient instance

        Returns:
            PMID if published version exists
        """
        if not pubmed_client:
            logger.debug("No PubMed client provided for published version check")
            return None

        # Search PubMed by DOI
        return pubmed_client.search_by_doi(doi)

    def is_biorxiv_url(self, url: str) -> bool:
        """Check if URL is from bioRxiv."""
        return 'biorxiv.org' in url.lower()

    def is_medrxiv_url(self, url: str) -> bool:
        """Check if URL is from medRxiv."""
        return 'medrxiv.org' in url.lower()

    def is_preprint_url(self, url: str) -> bool:
        """Check if URL is from bioRxiv or medRxiv."""
        return self.is_biorxiv_url(url) or self.is_medrxiv_url(url)
