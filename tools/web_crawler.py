"""
Web Crawler

Enhanced web content extraction with anti-bot handling and content routing.
Routes specialized URLs (PubMed, bioRxiv, ClinicalTrials) to appropriate clients.
"""

import asyncio
import re
from typing import Optional, Dict, Any, List
import httpx
import trafilatura

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class WebCrawler:
    """
    Web content crawler with anti-bot evasion and content routing.

    Features:
    - URL-type detection and routing to specialized clients
    - Browser header spoofing
    - Smart retry mechanism
    - Paywall detection with PubMed fallback
    - Content extraction via trafilatura
    """

    # Paywall detection keywords
    PAYWALL_INDICATORS = [
        'subscription required',
        'access denied',
        'login required',
        'purchase access',
        'institutional access',
        'paywall',
        'this article is part of the',
        'subscribe to view'
    ]

    def __init__(
        self,
        max_retries: int = 2,
        timeout: float = 12.0,
        min_content_length: int = 200,
        pubmed_client: Optional[Any] = None,
        biorxiv_client: Optional[Any] = None,
        clinical_trials_client: Optional[Any] = None,
        pdf_parser: Optional[Any] = None
    ):
        """
        Initialize web crawler with optional specialized clients.

        Args:
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            min_content_length: Minimum valid content length
            pubmed_client: PubMedClient for fallback abstract fetching
            biorxiv_client: BioRxivClient for preprint handling
            clinical_trials_client: ClinicalTrialsClient for clinical data
            pdf_parser: PDFParser for PDF content extraction
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_content_length = min_content_length

        # Specialized clients (injected at runtime)
        self.pubmed_client = pubmed_client
        self.biorxiv_client = biorxiv_client
        self.clinical_trials_client = clinical_trials_client
        self.pdf_parser = pdf_parser

        logger.info("Web crawler initialized", max_retries=max_retries, timeout=timeout)

    def set_clients(
        self,
        pubmed_client: Any = None,
        biorxiv_client: Any = None,
        clinical_trials_client: Any = None,
        pdf_parser: Any = None
    ):
        """
        Set specialized clients after initialization.

        Used by main.py to inject clients after they're all created.
        """
        if pubmed_client:
            self.pubmed_client = pubmed_client
        if biorxiv_client:
            self.biorxiv_client = biorxiv_client
        if clinical_trials_client:
            self.clinical_trials_client = clinical_trials_client
        if pdf_parser:
            self.pdf_parser = pdf_parser

    def _get_browser_headers(self, referer: str = "https://www.google.com/") -> Dict[str, str]:
        """
        Generate realistic browser headers.

        Args:
            referer: Referer URL

        Returns:
            Headers dictionary
        """
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': referer,
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Sec-Ch-Ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
        }

    async def crawl_content(
        self,
        url: str,
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Enhanced content crawling with URL-type routing.

        Routes to specialized clients based on URL type:
        - ClinicalTrials.gov → ClinicalTrialsClient
        - bioRxiv/medRxiv → BioRxivClient → PubMed fallback
        - PDF → PDFParser → PubMed fallback
        - Web pages → trafilatura → PubMed fallback

        Args:
            url: Target URL
            title: Optional title for PubMed fallback search

        Returns:
            Extracted content string, or None if all methods fail
        """
        logger.info("Crawl content with routing", url=url[:100])

        # === Priority 1: ClinicalTrials.gov ===
        nct_id = self._extract_nct_id(url)
        if nct_id and self.clinical_trials_client:
            logger.info("Detected ClinicalTrials.gov URL", nct_id=nct_id)
            try:
                ct_data = await self.clinical_trials_client.fetch_study(nct_id)
                if ct_data:
                    content = self.clinical_trials_client.format_content(ct_data)
                    if content:
                        logger.info("ClinicalTrials API success", length=len(content))
                        return content
            except Exception as e:
                logger.warning("ClinicalTrials API failed", error=str(e))

        # === Priority 1.5: bioRxiv/medRxiv ===
        biorxiv_doi = self._extract_biorxiv_doi(url)
        if biorxiv_doi:
            logger.info("Detected bioRxiv/medRxiv URL", doi=biorxiv_doi)

            # Try PubMed first (for published version)
            if self.pubmed_client:
                pmid = await self._pubmed_search_by_doi(biorxiv_doi)
                if pmid:
                    logger.info("bioRxiv found in PubMed", pmid=pmid)
                    abstract = await self._pubmed_fetch_abstract(pmid)
                    if abstract:
                        return self._format_pubmed_result(abstract, "bioRxiv->PubMed")

            # Try bioRxiv API
            if self.biorxiv_client:
                try:
                    biorxiv_data = await self.biorxiv_client.fetch_details(biorxiv_doi)
                    if biorxiv_data:
                        content = self.biorxiv_client.format_content(biorxiv_data)
                        if content:
                            logger.info("bioRxiv API success", length=len(content))
                            return content
                except Exception as e:
                    logger.warning("bioRxiv API failed", error=str(e))

            # If PDF link and bioRxiv API failed, continue to PDF handling
            if '.pdf' not in url.lower():
                logger.warning("bioRxiv/medRxiv content not available")
                return None

        # === Priority 2: PDF ===
        if self.is_pdf_url(url):
            logger.info("Detected PDF link", url=url[:80])
            if self.pdf_parser:
                try:
                    pdf_text = await self.pdf_parser.parse(url)
                    if pdf_text and len(pdf_text) > 200:
                        logger.info("PDF parsing success", length=len(pdf_text))
                        return pdf_text
                except Exception as e:
                    logger.warning("PDF parsing failed", error=str(e))

            # PDF failed, try PubMed fallback
            return await self._fallback_to_pubmed(title, url)

        # === Priority 3: General web crawling ===
        result = await self.crawl(url)

        if result and result.get('content'):
            return result['content']

        # Web crawl failed, try PubMed fallback
        error = result.get('error') if result else 'unknown'
        logger.warning("Web crawl failed", url=url[:80], error=error)
        return await self._fallback_to_pubmed(title, url)

    async def _fallback_to_pubmed(
        self,
        title: Optional[str],
        url: str
    ) -> Optional[str]:
        """
        Enhanced PubMed fallback strategy.

        Tries multiple strategies in order:
        1. Extract PMID from URL
        2. Search by title (exact)
        3. Search by cleaned title (fuzzy)
        4. Search by DOI from URL
        5. ResearchGate-specific title cleanup

        Args:
            title: Article title (optional)
            url: Original URL

        Returns:
            PubMed abstract content, or None
        """
        if not self.pubmed_client:
            logger.warning("PubMed client not available for fallback")
            return None

        logger.info("Starting PubMed fallback", url=url[:80])

        pmid = None
        method = None

        # Strategy 1: Extract PMID from URL
        pmid = self._extract_pmid_from_url(url)
        if pmid:
            method = "URL_EXTRACTION"
            logger.info("Fallback Strategy 1: PMID from URL", pmid=pmid)

        # Strategy 2: Exact title search
        if not pmid and title:
            logger.info("Fallback Strategy 2: Exact title search")
            pmid = await self._pubmed_search_by_title(title)
            if pmid:
                method = "TITLE_EXACT"

        # Strategy 3: Fuzzy title search (cleaned)
        if not pmid and title:
            cleaned_title = re.sub(r'[:\[\](){}]', ' ', title)
            cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()

            if cleaned_title != title:
                logger.info("Fallback Strategy 3: Fuzzy title search")
                pmid = await self._pubmed_search_by_title(cleaned_title)
                if pmid:
                    method = "TITLE_FUZZY"

        # Strategy 4: DOI search
        if not pmid:
            doi = self._extract_doi_from_url(url)
            if doi:
                logger.info("Fallback Strategy 4: DOI search", doi=doi)
                pmid = await self._pubmed_search_by_doi(doi)
                if pmid:
                    method = "DOI_SEARCH"

        # Strategy 5: ResearchGate special handling
        if not pmid and title and 'researchgate.net' in url:
            logger.info("Fallback Strategy 5: ResearchGate cleanup")
            clean_title = re.sub(r'^(The|A|An)\s+', '', title, flags=re.IGNORECASE)
            clean_title = re.sub(r'\s+(is|are|reveals?|shows?)\s+.+$', '', clean_title, flags=re.IGNORECASE)
            clean_title = re.sub(r'\s+(via|through|by|in|of|and)\s+', ' ', clean_title, flags=re.IGNORECASE)
            if len(clean_title) > 100:
                clean_title = clean_title[:100]
            clean_title = re.sub(r'\s+', ' ', clean_title).strip()

            pmid = await self._pubmed_search_by_title(clean_title)
            if pmid:
                method = "RESEARCHGATE_CLEANUP"

        # All strategies exhausted
        if not pmid:
            logger.warning("All PubMed fallback strategies failed")
            return None

        # Fetch abstract
        abstract = await self._pubmed_fetch_abstract(pmid)
        if not abstract:
            logger.warning("Could not fetch abstract for PMID", pmid=pmid)
            return None

        logger.info("PubMed fallback success", method=method, pmid=pmid)
        return self._format_pubmed_result(abstract, f"Fallback-{method}")

    def _format_pubmed_result(self, abstract_data: Dict[str, Any], source: str) -> str:
        """Format PubMed abstract data as readable text."""
        return f"""Title: {abstract_data.get('title', 'N/A')}

Authors: {abstract_data.get('authors', 'N/A')}

Journal: {abstract_data.get('journal', 'N/A')}

Date: {abstract_data.get('date', abstract_data.get('year', 'N/A'))}

PMID: {abstract_data.get('pmid', 'N/A')}

Abstract:
{abstract_data.get('abstract', 'No abstract available.')}

[Source: {source}]"""

    async def _pubmed_search_by_title(self, title: str) -> Optional[str]:
        """Run sync PubMed title search without blocking the event loop."""
        return await asyncio.to_thread(self.pubmed_client.search_by_title, title)

    async def _pubmed_search_by_doi(self, doi: str) -> Optional[str]:
        """Run sync PubMed DOI search without blocking the event loop."""
        return await asyncio.to_thread(self.pubmed_client.search_by_doi, doi)

    async def _pubmed_fetch_abstract(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Run sync PubMed abstract fetch without blocking the event loop."""
        return await asyncio.to_thread(self.pubmed_client.fetch_abstract, pmid)

    def _extract_nct_id(self, url: str) -> Optional[str]:
        """Extract NCT ID from ClinicalTrials.gov URL."""
        match = re.search(r'(NCT\d{8})', url, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _extract_biorxiv_doi(self, url: str) -> Optional[str]:
        """Extract DOI from bioRxiv/medRxiv URL."""
        if 'biorxiv.org' not in url and 'medrxiv.org' not in url:
            return None
        pattern = r'(10\.1101/(?:\d{4}\.\d{2}\.\d{2}\.)?\d+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None

    def _extract_pmid_from_url(self, url: str) -> Optional[str]:
        """Extract PMID from PubMed URL."""
        patterns = [
            r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)',
            r'ncbi\.nlm\.nih\.gov/pubmed/(\d+)',
            r'/pubmed/(\d+)',
            r'PMID[:\s]+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from URL."""
        # Check bioRxiv first
        biorxiv_doi = self._extract_biorxiv_doi(url)
        if biorxiv_doi:
            return biorxiv_doi

        patterns = [
            r'doi\.org/([\d\.]+/[^\s"<>]+)',
            r'/doi/([\d\.]+/[^\s"<>]+)',
            r'doi:([\d\.]+/[^\s"<>]+)',
            r'/articles/([sd]\d+)',  # Nature format
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                doi = match.group(1)
                # Nature format conversion
                if doi.startswith('s') or doi.startswith('d'):
                    doi = f"10.1038/{doi}"
                return doi
        return None

    async def crawl(
        self,
        url: str,
        extract_metadata: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Basic URL crawl and content extraction.

        Args:
            url: Target URL
            extract_metadata: Whether to extract title/author metadata

        Returns:
            Dictionary with 'content', 'title', 'status' keys
        """
        logger.debug("Crawling URL", url=url)

        result = {
            'url': url,
            'content': None,
            'title': None,
            'status': None,
            'error': None
        }

        # Attempt fetch with retries
        html_content = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True,
                    verify=False
                ) as client:
                    headers = self._get_browser_headers()
                    response = await client.get(url, headers=headers)

                    result['status'] = response.status_code

                    # Handle specific status codes
                    if response.status_code == 429:
                        # Rate limited - wait and retry
                        logger.warning("Rate limited", url=url, attempt=attempt + 1)
                        await asyncio.sleep(3)
                        continue

                    elif response.status_code in (401, 403):
                        # Access denied
                        logger.warning("Access denied", url=url, status=response.status_code)
                        result['error'] = 'access_denied'
                        return result

                    elif response.status_code != 200:
                        logger.warning("HTTP error", url=url, status=response.status_code)
                        last_error = f"HTTP {response.status_code}"
                        continue

                    html_content = response.text
                    break

            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning("Request timeout", url=url, attempt=attempt + 1)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

            except httpx.ConnectError as e:
                last_error = "connection_error"
                logger.warning("Connection error", url=url, error=str(e))
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

            except Exception as e:
                last_error = str(e)
                logger.error("Unexpected error", url=url, error=str(e))
                break

        if not html_content:
            result['error'] = last_error or 'fetch_failed'
            return result

        # Check for paywall before extraction
        if self._detect_paywall(html_content):
            logger.warning("Paywall detected", url=url)
            result['error'] = 'paywall'
            return result

        # Extract content using trafilatura
        try:
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True
            )

            if extracted and len(extracted) >= self.min_content_length:
                result['content'] = extracted

                # Extract metadata if requested
                if extract_metadata:
                    metadata = trafilatura.extract_metadata(html_content)
                    if metadata:
                        result['title'] = metadata.title
                        result['author'] = metadata.author
                        result['date'] = metadata.date

                logger.debug("Content extracted", url=url, length=len(extracted))
            else:
                result['error'] = 'content_too_short'
                logger.warning("Extracted content too short", url=url, length=len(extracted) if extracted else 0)

        except Exception as e:
            result['error'] = f"extraction_error: {str(e)}"
            logger.error("Content extraction failed", url=url, error=str(e))

        return result

    def _detect_paywall(self, html: str) -> bool:
        """
        Detect paywall in HTML content.

        Args:
            html: Raw HTML string

        Returns:
            True if paywall detected
        """
        html_lower = html.lower()
        for indicator in self.PAYWALL_INDICATORS:
            if indicator in html_lower:
                return True
        return False

    async def crawl_batch(
        self,
        urls: List[str],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs
            max_concurrent: Maximum concurrent requests

        Returns:
            List of crawl results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_crawl(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.crawl(url)

        tasks = [limited_crawl(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                processed_results.append({
                    'url': url,
                    'content': None,
                    'error': str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def extract_links(self, url: str, pattern: str = None) -> List[str]:
        """
        Extract links from a page.

        Args:
            url: Page URL
            pattern: Optional regex pattern to filter links

        Returns:
            List of URLs
        """
        from urllib.parse import urljoin

        result = await self.crawl(url)
        if not result or result.get('error'):
            return []

        # Get raw HTML for link extraction
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=self._get_browser_headers())
                html = response.text
            except:
                return []

        # Extract href attributes
        href_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(href_pattern, html)

        # Convert to absolute URLs
        links = []
        for match in matches:
            abs_url = urljoin(url, match)
            if abs_url.startswith('http'):
                if pattern is None or re.search(pattern, abs_url):
                    links.append(abs_url)

        return list(set(links))

    def is_pdf_url(self, url: str) -> bool:
        """
        Check if URL points to a PDF.

        Args:
            url: URL to check

        Returns:
            True if likely PDF
        """
        url_lower = url.lower()
        return (
            url_lower.endswith('.pdf') or
            '/pdf/' in url_lower or
            'pdf.php' in url_lower
        )

    def is_pubmed_url(self, url: str) -> bool:
        """Check if URL is PubMed."""
        return 'pubmed.ncbi.nlm.nih.gov' in url or 'ncbi.nlm.nih.gov/pubmed' in url

    def is_pmc_url(self, url: str) -> bool:
        """Check if URL is PMC."""
        return 'ncbi.nlm.nih.gov/pmc' in url or 'pmc.ncbi.nlm.nih.gov' in url

    def is_biorxiv_url(self, url: str) -> bool:
        """Check if URL is bioRxiv/medRxiv."""
        return 'biorxiv.org' in url or 'medrxiv.org' in url

    def is_clinical_trials_url(self, url: str) -> bool:
        """Check if URL is ClinicalTrials.gov."""
        return 'clinicaltrials.gov' in url
