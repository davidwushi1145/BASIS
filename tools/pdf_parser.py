"""
PDF Parser

Dual-layer PDF parsing with MinerU API + pdfplumber fallback.
"""

import io
import asyncio
import re
import time
import zipfile
import json
import tempfile
from typing import Optional, Dict, Any, List
from pathlib import Path
import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """
    PDF parsing with commercial API + open source fallback.

    Strategy 1: MinerU API (supports formulas, tables)
    Strategy 2: pdfplumber (local, reliable)
    """

    def __init__(
        self,
        mineru_token: str = None,
        cache_dir: str = None,
        max_pages: int = 50,
        max_file_size: int = 50 * 1024 * 1024  # 50 MB
    ):
        """
        Initialize PDF parser.

        Args:
            mineru_token: MinerU API token
            cache_dir: Directory for caching PDFs
            max_pages: Maximum pages to parse
            max_file_size: Maximum file size in bytes
        """
        self.mineru_token = mineru_token or getattr(settings, 'MINERU_API_TOKEN', None)
        self.cache_dir = Path(cache_dir or getattr(settings, 'PDF_CACHE_DIR', '/tmp/pdf_cache'))
        self.max_pages = max_pages
        self.max_file_size = max_file_size

        # MinerU API settings (ev1 parity: /api/v4/extract/task flow)
        self.mineru_api_base = getattr(settings, "MINERU_API_BASE", "https://mineru.net/api/v4/extract/task")
        self.mineru_model_version = getattr(settings, "MINERU_MODEL_VERSION", "vlm")
        self.mineru_poll_interval = getattr(settings, "MINERU_POLL_INTERVAL", 5)
        self.mineru_max_poll_time = getattr(settings, "MINERU_MAX_POLL_TIME", 300)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "PDF parser initialized",
            has_mineru=bool(self.mineru_token),
            cache_dir=str(self.cache_dir),
            mineru_api_base=self.mineru_api_base
        )

    async def parse(
        self,
        pdf_url: str,
        source_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Parse PDF from URL.

        Args:
            pdf_url: URL to PDF file
            source_url: Original article URL (for Referer header)

        Returns:
            Extracted text content or None
        """
        logger.debug("Parsing PDF", url=pdf_url)

        # Try MinerU first if available
        if self.mineru_token:
            try:
                content = await self._parse_mineru(pdf_url)
                if content and len(content) > 200:
                    return content
                logger.debug("MinerU returned insufficient content, falling back")
            except Exception as e:
                logger.warning("MinerU parsing failed", error=str(e))

        # Fallback to pdfplumber
        try:
            return await self._parse_pdfplumber(pdf_url, source_url)
        except Exception as e:
            logger.error("PDF parsing failed completely", error=str(e))
            return None

    async def parse_url(
        self,
        pdf_url: str,
        source_url: Optional[str] = None
    ) -> Optional[str]:
        """Compatibility wrapper for callers using legacy method name."""
        return await self.parse(pdf_url, source_url)

    async def _parse_mineru(self, pdf_url: str) -> Optional[str]:
        """
        Parse PDF using MinerU API (ev1-aligned /api/v4/extract/task).

        Args:
            pdf_url: URL to PDF

        Returns:
            Extracted content with tables/figures
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mineru_token}",
        }
        page_end = min(15, max(1, int(self.max_pages)))
        payload = {
            "url": pdf_url,
            "model_version": self.mineru_model_version,
            "page_ranges": f"1-{page_end}",
        }

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            # Step 1: Create task
            try:
                create_resp = await client.post(
                    self.mineru_api_base,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
            except Exception as e:
                logger.warning("MinerU task creation request failed", error=str(e))
                return None

            if create_resp.status_code != 200:
                logger.error(
                    "MinerU task creation failed",
                    status=create_resp.status_code,
                    body=create_resp.text[:200]
                )
                return None

            try:
                create_json = create_resp.json()
            except Exception:
                logger.error("MinerU task creation returned non-JSON response")
                return None

            if create_json.get("code") != 0:
                logger.error("MinerU API error on create", msg=create_json.get("msg"))
                return None

            task_id = (create_json.get("data") or {}).get("task_id")
            if not task_id:
                logger.error("MinerU create response missing task_id")
                return None

            logger.debug("MinerU task created", task_id=task_id)

            # Step 2: Poll for completion
            query_url = f"{self.mineru_api_base.rstrip('/')}/{task_id}"
            start_time = time.time()

            while time.time() - start_time < self.mineru_max_poll_time:
                try:
                    poll_resp = await client.get(query_url, headers=headers, timeout=30.0)
                except Exception as e:
                    logger.warning("MinerU polling request failed", error=str(e))
                    await asyncio.sleep(self.mineru_poll_interval)
                    continue

                if poll_resp.status_code != 200:
                    logger.warning("MinerU poll failed", status=poll_resp.status_code)
                    await asyncio.sleep(self.mineru_poll_interval)
                    continue

                try:
                    poll_json = poll_resp.json()
                except Exception:
                    logger.warning("MinerU poll returned non-JSON response")
                    await asyncio.sleep(self.mineru_poll_interval)
                    continue

                if poll_json.get("code") != 0:
                    logger.error("MinerU API error on poll", msg=poll_json.get("msg"))
                    return None

                data = poll_json.get("data") or {}
                state = data.get("state")
                if state == "done":
                    zip_url = data.get("full_zip_url") or data.get("zip_url") or data.get("download_url")
                    if not zip_url:
                        logger.error("MinerU done state missing zip URL", keys=list(data.keys()))
                        return None
                    return await self._download_mineru_result(client, zip_url, headers=headers)
                if state == "failed":
                    logger.error("MinerU task failed", err_msg=data.get("err_msg"))
                    return None

                await asyncio.sleep(self.mineru_poll_interval)

            logger.warning("MinerU polling timeout", task_id=task_id, timeout_s=self.mineru_max_poll_time)
            return None

    async def _download_mineru_result(
        self,
        client: httpx.AsyncClient,
        download_url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Download and extract MinerU result ZIP."""
        resp = await client.get(download_url, headers=headers, timeout=60.0)
        if resp.status_code != 200:
            return None

        # Extract ZIP in memory
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            content_parts = []

            # Read main markdown content
            names = set(zf.namelist())
            if 'full.md' in names:
                content_parts.append("[MAIN CONTENT]")
                content_parts.append(zf.read('full.md').decode('utf-8'))
            else:
                # Fallback: pick the largest .md file if full.md is not present.
                md_files = [n for n in names if n.lower().endswith(".md")]
                if md_files:
                    md_files.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                    content_parts.append("[MAIN CONTENT]")
                    content_parts.append(zf.read(md_files[0]).decode("utf-8", errors="replace"))

            # Parse content_list.json for tables/figures
            if 'content_list.json' in names:
                try:
                    content_list = json.loads(zf.read('content_list.json'))
                    tables, figures = self._extract_structured_content(content_list)

                    if tables:
                        content_parts.append("\n[TABLES]")
                        content_parts.extend(tables)

                    if figures:
                        content_parts.append("\n[FIGURES]")
                        content_parts.extend(figures)
                except:
                    pass

            return '\n'.join(content_parts)

    def _extract_structured_content(
        self,
        content_list: List[Dict]
    ) -> tuple[List[str], List[str]]:
        """Extract tables and figures from content_list."""
        tables = []
        figures = []

        for item in content_list:
            item_type = item.get('type', '')

            if item_type == 'table':
                table_md = item.get('markdown', item.get('content', ''))
                if table_md:
                    tables.append(table_md)

            elif item_type == 'figure':
                caption = item.get('caption', item.get('description', ''))
                if caption:
                    figures.append(f"Figure: {caption}")

        return tables, figures

    async def _parse_pdfplumber(
        self,
        pdf_url: str,
        source_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Parse PDF using pdfplumber (local).

        Args:
            pdf_url: URL to PDF
            source_url: Original URL for Referer header

        Returns:
            Extracted text content
        """
        import pdfplumber

        # Download PDF
        pdf_content = await self._download_pdf(pdf_url, source_url)
        if not pdf_content:
            return None

        # Parse with pdfplumber
        content_parts = []

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            page_count = min(len(pdf.pages), self.max_pages)

            for i, page in enumerate(pdf.pages[:page_count]):
                # Extract text
                text = page.extract_text()
                if text:
                    content_parts.append(f"[Page {i + 1}]")
                    content_parts.append(text)

                # Extract tables
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    md_table = self._table_to_markdown(table)
                    if md_table:
                        content_parts.append(f"\n[TABLE Page {i + 1}.{j + 1}]")
                        content_parts.append(md_table)

        if not content_parts:
            return None

        content = '\n'.join(content_parts)
        logger.debug("pdfplumber parsing completed", pages=page_count, length=len(content))
        return content

    async def _download_pdf(
        self,
        pdf_url: str,
        source_url: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Download PDF with proper headers.

        Args:
            pdf_url: URL to PDF
            source_url: Original URL for Referer

        Returns:
            PDF bytes or None
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,*/*',
        }

        # bioRxiv/medRxiv special handling
        if 'biorxiv.org' in pdf_url or 'medrxiv.org' in pdf_url:
            article_url = pdf_url.replace('.full.pdf', '').replace('.pdf', '')
            headers['Referer'] = source_url or article_url
            logger.debug("Using bioRxiv/medRxiv Referer bypass")

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                resp = await client.get(pdf_url, headers=headers)

                # Verify response
                if resp.status_code != 200:
                    logger.warning("PDF download failed", status=resp.status_code, url=pdf_url)
                    return None

                # Check content type
                content_type = resp.headers.get('content-type', '')
                if 'pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                    logger.warning("Invalid content type for PDF", content_type=content_type)
                    return None

                # Check file size
                if len(resp.content) > self.max_file_size:
                    logger.warning("PDF too large", size=len(resp.content))
                    return None

                return resp.content

            except Exception as e:
                logger.error("PDF download error", error=str(e))
                return None

    def _table_to_markdown(self, table: List[List]) -> str:
        """
        Convert pdfplumber table to Markdown format.

        Args:
            table: 2D list of cell values

        Returns:
            Markdown table string
        """
        if not table or not table[0]:
            return ""

        lines = []

        # Header row
        header = [str(cell or '').strip() for cell in table[0]]
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

        # Data rows
        for row in table[1:]:
            cells = [str(cell or '').strip() for cell in row]
            # Pad if needed
            while len(cells) < len(header):
                cells.append('')
            lines.append('| ' + ' | '.join(cells[:len(header)]) + ' |')

        return '\n'.join(lines)

    def parse_local(self, pdf_path: str) -> Optional[str]:
        """
        Parse local PDF file.

        Args:
            pdf_path: Path to local PDF

        Returns:
            Extracted text
        """
        import pdfplumber

        path = Path(pdf_path)
        if not path.exists():
            logger.error("PDF file not found", path=pdf_path)
            return None

        if path.stat().st_size > self.max_file_size:
            logger.error("PDF file too large", path=pdf_path)
            return None

        content_parts = []

        with pdfplumber.open(path) as pdf:
            page_count = min(len(pdf.pages), self.max_pages)

            for i, page in enumerate(pdf.pages[:page_count]):
                text = page.extract_text()
                if text:
                    content_parts.append(f"[Page {i + 1}]")
                    content_parts.append(text)

                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    md_table = self._table_to_markdown(table)
                    if md_table:
                        content_parts.append(f"\n[TABLE Page {i + 1}.{j + 1}]")
                        content_parts.append(md_table)

        return '\n'.join(content_parts) if content_parts else None


# Need asyncio for sleep
import asyncio
