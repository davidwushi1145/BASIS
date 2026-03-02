"""
DepMap Knowledge Base Retriever

Retriever for DepMap gene dependency, essentiality data, and SL benchmark pairs.
Features:
- CRISPR gene effect scores
- SL benchmark pair validation
- Synthetic document generation for retrieval augmentation
"""

import csv
import os
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from retrieval.base import BaseRetriever, RetrievalResult, SourceType
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DepMapKBRetriever(BaseRetriever):
    """
    Retriever for DepMap (Cancer Dependency Map) data.

    Data includes:
    - Gene essentiality scores across cell lines
    - CRISPR knockout effects
    - Cancer type-specific dependencies
    - Synthetic Lethality (SL) benchmark pairs
    """

    def __init__(
        self,
        data_dir: str = None,
        source_type: SourceType = SourceType.DEPMAP_KB
    ):
        """
        Initialize DepMap retriever.

        Args:
            data_dir: Directory containing DepMap CSV files
            source_type: Source type for results
        """
        super().__init__(source_type)
        self.data_dir = Path(data_dir or getattr(settings, 'DEPMAP_DATA_DIR', './data/depmap'))

        # In-memory data stores
        self._gene_effects: Dict[str, Dict[str, float]] = {}  # gene -> {cell_line: score}
        self._cell_line_info: Dict[str, Dict[str, str]] = {}  # cell_line -> {cancer_type, ...}
        self._gene_summaries: Dict[str, Dict[str, Any]] = {}  # gene -> aggregated stats

        # SL Benchmark data
        self._sl_benchmark = None  # DataFrame if pandas available
        self._sl_pairs_cache: Set[tuple] = set()  # {(gene_a, gene_b)} for fast lookup

        self._loaded = False
        self._load_data()

    def _load_data(self):
        """Load DepMap data from CSV files."""
        if not self.data_dir.exists():
            logger.warning("DepMap data directory not found", path=str(self.data_dir))
            return

        try:
            # Load gene effect scores
            effect_file = self.data_dir / "CRISPRGeneEffect.csv"
            if effect_file.exists():
                self._load_gene_effects(effect_file)

            # Load cell line info
            cell_line_file = self.data_dir / "Model.csv"
            if cell_line_file.exists():
                self._load_cell_line_info(cell_line_file)

            # Compute gene summaries
            self._compute_gene_summaries()

            # Load SL Benchmark (validated SL pairs)
            self._load_sl_benchmark()

            self._loaded = True
            logger.info(
                "DepMap data loaded",
                genes=len(self._gene_effects),
                cell_lines=len(self._cell_line_info),
                sl_pairs=len(self._sl_pairs_cache)
            )

        except Exception as e:
            logger.error("Failed to load DepMap data", error=str(e))

    def _load_sl_benchmark(self):
        """Load SL Benchmark pairs from CSV."""
        # Try multiple possible filenames
        possible_files = [
            self.data_dir / "SL_Benchmark_Final.csv",
            self.data_dir / "sl_benchmark.csv",
            self.data_dir / "SL_pairs.csv"
        ]

        benchmark_file = None
        for f in possible_files:
            if f.exists():
                benchmark_file = f
                break

        if not benchmark_file:
            logger.debug("SL Benchmark file not found")
            return

        if not PANDAS_AVAILABLE:
            # Fallback to csv module
            try:
                with open(benchmark_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gene_a = str(row.get('gene_a', row.get('Gene_A', ''))).upper().strip()
                        gene_b = str(row.get('gene_b', row.get('Gene_B', ''))).upper().strip()
                        if gene_a and gene_b:
                            self._sl_pairs_cache.add((gene_a, gene_b))
                            self._sl_pairs_cache.add((gene_b, gene_a))
                logger.info("SL Benchmark loaded (csv fallback)", pairs=len(self._sl_pairs_cache) // 2)
            except Exception as e:
                logger.warning("Failed to load SL benchmark", error=str(e))
            return

        try:
            self._sl_benchmark = pd.read_csv(benchmark_file)

            # Build fast lookup cache (bidirectional)
            for _, row in self._sl_benchmark.iterrows():
                gene_a = str(row.get('gene_a', row.get('Gene_A', ''))).upper().strip()
                gene_b = str(row.get('gene_b', row.get('Gene_B', ''))).upper().strip()
                if gene_a and gene_b:
                    self._sl_pairs_cache.add((gene_a, gene_b))
                    self._sl_pairs_cache.add((gene_b, gene_a))

            logger.info("SL Benchmark loaded", pairs=len(self._sl_benchmark))

        except Exception as e:
            logger.warning("Failed to load SL benchmark", error=str(e))

    def _load_gene_effects(self, filepath: Path):
        """Load CRISPR gene effect scores."""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cell_line = row.get('DepMap_ID', row.get('', ''))

                for key, value in row.items():
                    if key in ('DepMap_ID', ''):
                        continue

                    # Parse gene name from column header (format: "GENE (ENTREZ)")
                    gene = key.split(' ')[0].strip()

                    try:
                        score = float(value) if value else None
                        if score is not None:
                            if gene not in self._gene_effects:
                                self._gene_effects[gene] = {}
                            self._gene_effects[gene][cell_line] = score
                    except ValueError:
                        pass

    def _load_cell_line_info(self, filepath: Path):
        """Load cell line metadata."""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cell_line_id = row.get('ModelID', row.get('DepMap_ID', ''))
                if cell_line_id:
                    self._cell_line_info[cell_line_id] = {
                        'name': row.get('CellLineName', row.get('StrippedCellLineName', '')),
                        'cancer_type': row.get('OncotreeLineage', row.get('PrimaryOrMetastasis', '')),
                        'subtype': row.get('OncotreeSubtype', ''),
                        'tissue': row.get('Tissue', row.get('OncotreePrimaryDisease', ''))
                    }

    def _compute_gene_summaries(self):
        """Compute aggregated statistics for each gene."""
        for gene, effects in self._gene_effects.items():
            scores = list(effects.values())
            if not scores:
                continue

            # Compute statistics
            mean_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            # Count essential cell lines (score < -0.5)
            essential_count = sum(1 for s in scores if s < -0.5)
            essential_pct = essential_count / len(scores) * 100

            # Identify cancer types where gene is essential
            essential_cancers = defaultdict(int)
            for cell_line, score in effects.items():
                if score < -0.5:
                    info = self._cell_line_info.get(cell_line, {})
                    cancer = info.get('cancer_type', 'Unknown')
                    essential_cancers[cancer] += 1

            self._gene_summaries[gene] = {
                'mean_score': mean_score,
                'min_score': min_score,
                'max_score': max_score,
                'essential_count': essential_count,
                'essential_pct': essential_pct,
                'total_cell_lines': len(scores),
                'essential_cancers': dict(essential_cancers)
            }

    async def search(
        self,
        query: str,
        top_k: int = 10,
        genes: List[str] = None,
        cancer_type: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search DepMap data.

        Args:
            query: Search query
            top_k: Maximum results
            genes: Specific genes to look up
            cancer_type: Filter by cancer type

        Returns:
            List of RetrievalResult
        """
        if not self._loaded:
            return []

        results = []

        # Search by genes
        if genes:
            for gene in genes[:top_k]:
                gene_upper = gene.upper()
                if gene_upper in self._gene_summaries:
                    content = self._format_gene_summary(gene_upper, cancer_type)
                    summary = self._gene_summaries[gene_upper]
                    results.append(RetrievalResult(
                        content=content,
                        score=abs(summary['mean_score']),  # Use essentiality as score
                        source=self.source_type,
                        metadata={
                            'gene': gene_upper,
                            'essential_pct': summary['essential_pct'],
                            'type': 'depmap_summary'
                        }
                    ))

        # If no genes provided, try to extract from query
        if not genes:
            # Simple extraction of potential gene names
            import re
            potential_genes = re.findall(r'\b[A-Z][A-Z0-9]{1,9}\b', query)
            for gene in potential_genes[:top_k]:
                if gene in self._gene_summaries:
                    content = self._format_gene_summary(gene, cancer_type)
                    summary = self._gene_summaries[gene]
                    results.append(RetrievalResult(
                        content=content,
                        score=abs(summary['mean_score']),
                        source=self.source_type,
                        metadata={'gene': gene, 'type': 'depmap_summary'}
                    ))

        logger.debug("DepMap search completed", query=query[:50], results=len(results))
        return results[:top_k]

    def _format_gene_summary(self, gene: str, cancer_type: str = None) -> str:
        """Format gene summary as text."""
        summary = self._gene_summaries.get(gene)
        if not summary:
            return f"No DepMap data available for {gene}"

        lines = [
            f"DepMap Gene Essentiality: {gene}",
            f"Mean CRISPR Effect: {summary['mean_score']:.3f}",
            f"Range: {summary['min_score']:.3f} to {summary['max_score']:.3f}",
            f"Essential in: {summary['essential_count']}/{summary['total_cell_lines']} cell lines ({summary['essential_pct']:.1f}%)",
        ]

        # Add cancer-specific info
        if summary['essential_cancers']:
            lines.append("")
            lines.append("Essential in cancer types:")
            sorted_cancers = sorted(
                summary['essential_cancers'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for cancer, count in sorted_cancers[:5]:
                lines.append(f"  - {cancer}: {count} cell lines")

        lines.append("")
        lines.append("[Source: DepMap CRISPR Gene Effect]")

        return '\n'.join(lines)

    async def get_gene_essentiality(
        self,
        gene: str,
        cancer_type: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed essentiality info for a gene.

        Args:
            gene: Gene symbol
            cancer_type: Optional cancer type filter

        Returns:
            Essentiality statistics
        """
        gene_upper = gene.upper()
        if gene_upper not in self._gene_summaries:
            return None

        summary = self._gene_summaries[gene_upper].copy()

        # Filter by cancer type if specified
        if cancer_type and gene_upper in self._gene_effects:
            cancer_scores = []
            for cell_line, score in self._gene_effects[gene_upper].items():
                info = self._cell_line_info.get(cell_line, {})
                if cancer_type.lower() in info.get('cancer_type', '').lower():
                    cancer_scores.append(score)

            if cancer_scores:
                summary['cancer_specific'] = {
                    'cancer_type': cancer_type,
                    'mean_score': sum(cancer_scores) / len(cancer_scores),
                    'essential_count': sum(1 for s in cancer_scores if s < -0.5),
                    'total': len(cancer_scores)
                }

        return summary

    async def get_essential_genes(
        self,
        cancer_type: str = None,
        threshold: float = -0.5,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get most essential genes, optionally filtered by cancer type.

        Args:
            cancer_type: Cancer type filter
            threshold: Essentiality threshold
            top_k: Number of genes to return

        Returns:
            List of essential genes with scores
        """
        essential_genes = []

        for gene, summary in self._gene_summaries.items():
            if summary['mean_score'] < threshold:
                essential_genes.append({
                    'gene': gene,
                    'mean_score': summary['mean_score'],
                    'essential_pct': summary['essential_pct']
                })

        # Sort by mean score (most negative = most essential)
        essential_genes.sort(key=lambda x: x['mean_score'])

        return essential_genes[:top_k]

    async def compare_genes(
        self,
        gene1: str,
        gene2: str
    ) -> Dict[str, Any]:
        """
        Compare essentiality of two genes.

        Args:
            gene1: First gene
            gene2: Second gene

        Returns:
            Comparison data
        """
        summary1 = self._gene_summaries.get(gene1.upper())
        summary2 = self._gene_summaries.get(gene2.upper())

        return {
            'gene1': {
                'symbol': gene1.upper(),
                'data': summary1
            },
            'gene2': {
                'symbol': gene2.upper(),
                'data': summary2
            },
            'correlation': self._compute_correlation(gene1.upper(), gene2.upper())
        }

    def _compute_correlation(self, gene1: str, gene2: str) -> Optional[float]:
        """Compute correlation between two genes' effects."""
        effects1 = self._gene_effects.get(gene1, {})
        effects2 = self._gene_effects.get(gene2, {})

        # Find common cell lines
        common_lines = set(effects1.keys()) & set(effects2.keys())
        if len(common_lines) < 10:
            return None

        scores1 = [effects1[cl] for cl in common_lines]
        scores2 = [effects2[cl] for cl in common_lines]

        # Pearson correlation
        n = len(scores1)
        mean1 = sum(scores1) / n
        mean2 = sum(scores2) / n

        numerator = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2))
        denom1 = sum((s - mean1) ** 2 for s in scores1) ** 0.5
        denom2 = sum((s - mean2) ** 2 for s in scores2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            return None

        return numerator / (denom1 * denom2)

    def is_available(self) -> bool:
        """Check if retriever is available."""
        return self._loaded

    # ==================== SL Pair Methods ====================

    def get_sl_pairs(self, gene: str) -> List[Dict[str, Any]]:
        """
        Get known synthetic lethal partners for a gene.

        Args:
            gene: Gene symbol

        Returns:
            List of dicts with partner info
        """
        if self._sl_benchmark is None or not PANDAS_AVAILABLE:
            # Fallback: just return partners from cache
            gene_upper = gene.upper()
            partners = []
            for pair in self._sl_pairs_cache:
                if pair[0] == gene_upper:
                    partners.append({
                        'partner': pair[1],
                        'evidence_source': 'Unknown',
                        'cell_line': 'N/A',
                        'pubmed_id': '',
                        'cancer_type': '',
                        'benchmark_level': ''
                    })
            return partners

        gene_upper = gene.upper()
        results = []

        # Find all pairs containing this gene
        matched = self._sl_benchmark[
            (self._sl_benchmark['gene_a'].str.upper() == gene_upper) |
            (self._sl_benchmark['gene_b'].str.upper() == gene_upper)
        ]

        for _, row in matched.iterrows():
            gene_a = str(row.get('gene_a', row.get('Gene_A', '')))
            gene_b = str(row.get('gene_b', row.get('Gene_B', '')))
            partner = gene_b if gene_a.upper() == gene_upper else gene_a

            results.append({
                'partner': partner,
                'evidence_source': row.get('evidence_source', row.get('Source', 'Unknown')),
                'cell_line': row.get('cell_line', row.get('Cell_Line', 'N/A')),
                'pubmed_id': row.get('pubmed_id', row.get('PubMed_ID', '')),
                'cancer_type': row.get('cancer_type', row.get('Cancer_Type', '')),
                'benchmark_level': row.get('benchmark_level', row.get('Level', ''))
            })

        return results

    def validate_sl_pair(self, gene_a: str, gene_b: str) -> Dict[str, Any]:
        """
        Validate whether two genes form a known SL pair.

        Args:
            gene_a: First gene
            gene_b: Second gene

        Returns:
            Validation result with evidence
        """
        gene_a_upper = gene_a.upper()
        gene_b_upper = gene_b.upper()

        # Check cache
        if (gene_a_upper, gene_b_upper) in self._sl_pairs_cache or \
           (gene_b_upper, gene_a_upper) in self._sl_pairs_cache:
            # Get detailed evidence
            pairs = self.get_sl_pairs(gene_a_upper)
            for p in pairs:
                if p['partner'].upper() == gene_b_upper:
                    return {
                        'is_validated': True,
                        'evidence': f"{p['evidence_source']} ({p['benchmark_level']})",
                        'confidence': 0.9 if 'Gold' in str(p['benchmark_level']) else 0.7,
                        'pubmed_id': p['pubmed_id']
                    }

            # Found in cache but no detailed info
            return {
                'is_validated': True,
                'evidence': 'SL Benchmark',
                'confidence': 0.7,
                'pubmed_id': ''
            }

        return {'is_validated': False, 'evidence': None, 'confidence': 0.0}

    # ==================== Synthetic Document Generation ====================

    def augment_with_depmap(self, genes: List[str]) -> List[Dict[str, Any]]:
        """
        Generate synthetic documents for genes to inject into retrieval.

        This creates structured documents summarizing DepMap data for genes,
        which can be mixed with literature results to improve coverage.

        Args:
            genes: List of gene symbols

        Returns:
            List of synthetic documents with 'content' and 'metadata' keys
        """
        if not settings.USE_DEPMAP_KB or not genes:
            return []

        synthetic_docs = []

        for gene in genes:
            gene_upper = gene.upper()

            # 1. Generate SL pairs document
            sl_pairs = self.get_sl_pairs(gene_upper)
            if sl_pairs:
                partners = [p['partner'] for p in sl_pairs[:5]]
                evidence_sources = set(p['evidence_source'] for p in sl_pairs if p['evidence_source'])

                content = f"[DepMap Knowledge Base - Synthetic Lethality]\n\n"
                content += f"Gene: {gene_upper}\n"
                content += f"Known Synthetic Lethal Partners: {', '.join(partners)}\n\n"

                for pair in sl_pairs[:3]:
                    content += f"- {gene_upper} is synthetic lethal with {pair['partner']}\n"
                    content += f"  Evidence Source: {pair['evidence_source']}\n"
                    if pair['cell_line'] and pair['cell_line'] != 'N/A':
                        content += f"  Cell Line: {pair['cell_line']}\n"
                    if pair['pubmed_id']:
                        content += f"  PubMed ID: {pair['pubmed_id']}\n"
                    content += "\n"

                content += f"Total validated SL interactions: {len(sl_pairs)}\n"
                if evidence_sources:
                    content += f"Data sources: {', '.join(evidence_sources)}"

                synthetic_docs.append({
                    'content': content,
                    'metadata': {
                        'paper_title': f"DepMap SL Database: {gene_upper} Interactions",
                        'link': f"depmap://sl_benchmark/{gene_upper}",
                        'source': 'DepMap_KB',
                        'date': '2024',
                        'is_synthetic': True,
                        'key_genes': [gene_upper] + partners[:3],
                        'key_methods': ['synthetic_lethality', 'depmap'],
                        'is_web': False
                    }
                })

            # 2. Generate essentiality document if data available
            if gene_upper in self._gene_summaries:
                summary = self._gene_summaries[gene_upper]
                content = f"[DepMap Knowledge Base - Gene Essentiality]\n\n"
                content += f"Gene: {gene_upper}\n"
                content += f"Mean CRISPR Effect Score: {summary['mean_score']:.3f}\n"
                content += f"Range: {summary['min_score']:.3f} to {summary['max_score']:.3f}\n"
                content += f"Essential in: {summary['essential_count']}/{summary['total_cell_lines']} "
                content += f"cell lines ({summary['essential_pct']:.1f}%)\n\n"

                if summary['essential_cancers']:
                    content += "Cancer types with highest essentiality:\n"
                    sorted_cancers = sorted(
                        summary['essential_cancers'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for cancer, count in sorted_cancers[:5]:
                        content += f"  - {cancer}: {count} cell lines\n"

                content += "\n[Source: DepMap CRISPR Gene Effect]"

                synthetic_docs.append({
                    'content': content,
                    'metadata': {
                        'paper_title': f"DepMap Gene Essentiality: {gene_upper}",
                        'link': f"depmap://essentiality/{gene_upper}",
                        'source': 'DepMap_KB',
                        'date': '2024',
                        'is_synthetic': True,
                        'key_genes': [gene_upper],
                        'key_methods': ['crispr', 'essentiality', 'depmap'],
                        'is_web': False
                    }
                })

        if synthetic_docs:
            logger.info("Generated synthetic documents", count=len(synthetic_docs))

        return synthetic_docs

    def format_kb_context(self, genes: List[str], max_tokens: int = None) -> str:
        """
        Format DepMap knowledge as context for LLM.

        Args:
            genes: Genes to include
            max_tokens: Maximum tokens

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or settings.TOKEN_BUDGET_KB_MAX

        lines = ["=== [DepMap Structured Knowledge (P1)] ==="]
        lines.append("Confidence Level: P1 (High - Validated Database)")
        lines.append("")

        char_count = 0
        max_chars = max_tokens * 4  # Approximate

        for gene in genes:
            gene_upper = gene.upper()

            # Add SL pairs info
            sl_pairs = self.get_sl_pairs(gene_upper)
            if sl_pairs:
                section = f"SL Partners for {gene_upper}: "
                section += ", ".join(p['partner'] for p in sl_pairs[:5])
                if len(sl_pairs) > 5:
                    section += f" (+{len(sl_pairs)-5} more)"
                lines.append(section)
                char_count += len(section)

            # Add essentiality info
            if gene_upper in self._gene_summaries:
                summary = self._gene_summaries[gene_upper]
                section = f"{gene_upper} Essentiality: "
                section += f"Mean={summary['mean_score']:.2f}, "
                section += f"Essential in {summary['essential_pct']:.0f}% of cell lines"
                lines.append(section)
                char_count += len(section)

            if char_count > max_chars:
                lines.append("[...truncated]")
                break

        lines.append("")
        lines.append("[Source: DepMap Knowledge Base]")

        return '\n'.join(lines)
