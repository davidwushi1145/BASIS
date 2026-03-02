"""
PubMed Client

5-level fallback chain for PubMed abstract retrieval.
"""

import re
import time
from typing import Optional, Dict, Any, List
from Bio import Entrez

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PubMedClient:
    """
    PubMed API client with 5-level fallback strategy.

    Fallback chain:
    1. URL PMID extraction
    2. Exact title search
    3. Fuzzy title search
    4. DOI to PMID
    5. ResearchGate special handling
    """

    def __init__(
        self,
        email: str = None,
        api_key: str = None,
        tool_name: str = "SLAgent_RAG"
    ):
        """
        Initialize PubMed client.

        Args:
            email: Email for NCBI API (required)
            api_key: Optional API key for higher rate limits
            tool_name: Tool identifier for NCBI
        """
        self.email = email or settings.NCBI_EMAIL
        self.api_key = api_key or getattr(settings, 'NCBI_API_KEY', None)
        self.tool_name = tool_name

        # Configure Entrez
        Entrez.email = self.email
        Entrez.tool = self.tool_name
        if self.api_key:
            Entrez.api_key = self.api_key

        # Rate limiting: 3 req/s without key, 10 req/s with key
        self.delay = 0.34 if self.api_key else 0.5

        logger.info("PubMed client initialized", email=self.email, has_api_key=bool(self.api_key))

    def fetch_abstract(self, pmid: str) -> Optional[Dict[str, Any]]:
        """
        Fetch abstract and metadata by PMID.

        Args:
            pmid: PubMed ID

        Returns:
            Dictionary with title, abstract, authors, journal, year, pmid
        """
        try:
            time.sleep(self.delay)

            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="xml",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            if not records.get('PubmedArticle'):
                logger.warning("No article found for PMID", pmid=pmid)
                return None

            article = records['PubmedArticle'][0]['MedlineCitation']['Article']

            # Extract abstract
            abstract = self._extract_abstract(article)

            # Extract authors
            authors = self._extract_authors(article)

            # Extract metadata
            title = str(article.get('ArticleTitle', ''))
            journal = article.get('Journal', {}).get('Title', '')
            year = self._extract_year(article)

            result = {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'year': year
            }

            logger.debug("PubMed abstract fetched", pmid=pmid, title=title[:50])
            return result

        except Exception as e:
            logger.error("Failed to fetch PubMed abstract", pmid=pmid, error=str(e))
            return None

    def _extract_abstract(self, article: Dict) -> str:
        """Extract and format abstract from article."""
        if 'Abstract' not in article:
            return ""

        abstract_texts = article['Abstract'].get('AbstractText', [])

        if isinstance(abstract_texts, list):
            parts = []
            for section in abstract_texts:
                if hasattr(section, 'attributes') and 'Label' in section.attributes:
                    label = section.attributes['Label']
                    parts.append(f"{label}: {str(section)}")
                else:
                    parts.append(str(section))
            return ' '.join(parts)
        else:
            return str(abstract_texts)

    def _extract_authors(self, article: Dict) -> str:
        """Extract and format author list."""
        if 'AuthorList' not in article:
            return ""

        authors = []
        for author in article['AuthorList']:
            if 'LastName' in author and 'Initials' in author:
                authors.append(f"{author['LastName']} {author['Initials']}")
            elif 'CollectiveName' in author:
                authors.append(author['CollectiveName'])

        # Limit to first 5 authors
        authors_str = ', '.join(authors[:5])
        if len(article.get('AuthorList', [])) > 5:
            authors_str += ' et al.'

        return authors_str

    def _extract_year(self, article: Dict) -> str:
        """Extract publication year."""
        try:
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            return pub_date.get('Year', '')
        except:
            return ""

    def search_by_title(self, title: str, exact: bool = True) -> Optional[str]:
        """
        Search PubMed by title.

        Args:
            title: Article title
            exact: If True, use exact phrase matching

        Returns:
            PMID if found, None otherwise
        """
        try:
            time.sleep(self.delay)

            if exact:
                query = f'"{title}"[Title]'
            else:
                # Clean title for fuzzy search
                cleaned = re.sub(r'[:\[\](){}]', ' ', title)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                query = f'{cleaned}[Title]'

            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=3,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()

            id_list = record.get('IdList', [])
            if id_list:
                pmid = id_list[0]
                logger.debug("Title search successful", title=title[:50], pmid=pmid, exact=exact)
                return pmid

            return None

        except Exception as e:
            logger.error("Title search failed", title=title[:50], error=str(e))
            return None

    def search_by_doi(self, doi: str) -> Optional[str]:
        """
        Convert DOI to PMID.

        Args:
            doi: DOI string

        Returns:
            PMID if found
        """
        try:
            time.sleep(self.delay)

            handle = Entrez.esearch(
                db="pubmed",
                term=f"{doi}[DOI]",
                retmax=1
            )
            record = Entrez.read(handle)
            handle.close()

            id_list = record.get('IdList', [])
            if id_list:
                pmid = id_list[0]
                logger.debug("DOI to PMID conversion successful", doi=doi, pmid=pmid)
                return pmid

            return None

        except Exception as e:
            logger.error("DOI search failed", doi=doi, error=str(e))
            return None

    def pmc_to_pmid(self, pmc_id: str) -> Optional[str]:
        """
        Convert PMC ID to PMID.

        Args:
            pmc_id: PMC ID (e.g., "PMC9279849" or "9279849")

        Returns:
            PMID if found
        """
        try:
            # Extract numeric part only
            pmc_match = re.search(r'PMC(\d+)', pmc_id, re.IGNORECASE)
            if pmc_match:
                pmc_uid = pmc_match.group(1)
            else:
                pmc_uid = pmc_id

            time.sleep(self.delay)

            handle = Entrez.elink(
                dbfrom="pmc",
                db="pubmed",
                id=pmc_uid,
                linkname="pmc_pubmed"
            )
            record = Entrez.read(handle)
            handle.close()

            # Navigate nested structure
            if (record and
                record[0].get('LinkSetDb') and
                len(record[0]['LinkSetDb']) > 0 and
                record[0]['LinkSetDb'][0].get('Link')):
                pmid = record[0]['LinkSetDb'][0]['Link'][0]['Id']
                logger.debug("PMC to PMID conversion successful", pmc_id=pmc_id, pmid=pmid)
                return pmid

            return None

        except Exception as e:
            logger.error("PMC to PMID conversion failed", pmc_id=pmc_id, error=str(e))
            return None

    def extract_pmid_from_url(self, url: str) -> Optional[str]:
        """
        Extract PMID from URL.

        Supports:
        - https://pubmed.ncbi.nlm.nih.gov/12345678
        - https://www.ncbi.nlm.nih.gov/pubmed/12345678
        - PMC articles (converted via API)

        Args:
            url: URL string

        Returns:
            PMID if found
        """
        # Direct PMID pattern
        pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', url)
        if pmid_match:
            return pmid_match.group(1)

        pmid_match = re.search(r'ncbi\.nlm\.nih\.gov/pubmed/(\d+)', url)
        if pmid_match:
            return pmid_match.group(1)

        # PMC pattern - needs conversion
        pmc_match = re.search(r'PMC(\d+)', url, re.IGNORECASE)
        if pmc_match:
            return self.pmc_to_pmid(pmc_match.group(0))

        return None

    def fallback_enhanced(
        self,
        title: Optional[str] = None,
        url: Optional[str] = None,
        doi: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        5-level fallback chain for retrieving PubMed content.

        Args:
            title: Article title
            url: Source URL
            doi: DOI if available

        Returns:
            Article data if found via any strategy
        """
        pmid = None

        # Strategy 1: Extract PMID from URL
        if url:
            try:
                pmid = self.extract_pmid_from_url(url)
                if pmid:
                    logger.debug("Strategy 1 (URL extraction) succeeded", pmid=pmid)
            except Exception as e:
                logger.debug("Strategy 1 failed", error=str(e))

        # Strategy 2: Exact title search
        if not pmid and title:
            try:
                pmid = self.search_by_title(title, exact=True)
                if pmid:
                    logger.debug("Strategy 2 (exact title) succeeded", pmid=pmid)
            except Exception as e:
                logger.debug("Strategy 2 failed", error=str(e))

        # Strategy 3: Fuzzy title search
        if not pmid and title:
            try:
                cleaned_title = re.sub(r'[:\[\](){}]', ' ', title)
                cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
                if cleaned_title != title:  # Only if cleaning changed something
                    pmid = self.search_by_title(cleaned_title, exact=False)
                    if pmid:
                        logger.debug("Strategy 3 (fuzzy title) succeeded", pmid=pmid)
            except Exception as e:
                logger.debug("Strategy 3 failed", error=str(e))

        # Strategy 4: DOI to PMID
        if not pmid and doi:
            try:
                pmid = self.search_by_doi(doi)
                if pmid:
                    logger.debug("Strategy 4 (DOI) succeeded", pmid=pmid)
            except Exception as e:
                logger.debug("Strategy 4 failed", error=str(e))

        # Strategy 5: ResearchGate special handling
        if not pmid and title and url and 'researchgate' in url.lower():
            try:
                # Extra cleaning for ResearchGate titles
                clean_title = re.sub(r'^(The|A|An)\s+', '', title)
                clean_title = re.sub(r'\s+(is|are|reveals?|shows?)\s+.+$', '', clean_title)
                clean_title = re.sub(r'\s+(via|through|by|in|of|and)\s+', ' ', clean_title)
                if len(clean_title) > 100:
                    clean_title = clean_title[:100]

                pmid = self.search_by_title(clean_title, exact=False)
                if pmid:
                    logger.debug("Strategy 5 (ResearchGate) succeeded", pmid=pmid)
            except Exception as e:
                logger.debug("Strategy 5 failed", error=str(e))

        # Fetch full abstract if PMID found
        if pmid:
            return self.fetch_abstract(pmid)

        logger.warning("All PubMed fallback strategies failed", title=title[:50] if title else None)
        return None

    def batch_fetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Batch fetch multiple abstracts.

        Args:
            pmids: List of PMIDs

        Returns:
            List of article data dictionaries
        """
        results = []
        for pmid in pmids:
            result = self.fetch_abstract(pmid)
            if result:
                results.append(result)
        return results

    def format_citation(self, article: Dict[str, Any]) -> str:
        """
        Format article as citation string.

        Args:
            article: Article data from fetch_abstract

        Returns:
            Formatted citation string
        """
        parts = []

        if article.get('authors'):
            parts.append(article['authors'])

        if article.get('title'):
            parts.append(article['title'])

        if article.get('journal'):
            parts.append(article['journal'])

        if article.get('year'):
            parts.append(f"({article['year']})")

        if article.get('pmid'):
            parts.append(f"PMID: {article['pmid']}")

        return '. '.join(parts)

    def format_full_content(self, article: Dict[str, Any]) -> str:
        """
        Format article as full content string for RAG.

        Args:
            article: Article data from fetch_abstract

        Returns:
            Formatted content string
        """
        sections = []

        sections.append(f"Title: {article.get('title', 'Unknown')}")

        if article.get('authors'):
            sections.append(f"Authors: {article['authors']}")

        if article.get('journal'):
            sections.append(f"Journal: {article['journal']} ({article.get('year', 'N/A')})")

        sections.append(f"PMID: {article.get('pmid', 'N/A')}")

        if article.get('abstract'):
            sections.append(f"\nAbstract:\n{article['abstract']}")

        sections.append("\n[Source: PubMed]")

        return '\n'.join(sections)
