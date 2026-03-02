"""
Entity Analyzer for Biomedical Queries

Combines rule-based and NER-based entity extraction with gene synonym expansion.
"""

import re
import httpx
import asyncio
from typing import Dict, Set, List, Optional, Any
from collections import OrderedDict

from config import settings
from utils.logger import get_logger
from utils.cache import LRUCache

logger = get_logger(__name__)


class EntityAnalyzer:
    """
    Biomedical entity extraction and gene synonym expansion.

    Features:
    - Rule-based gene name extraction
    - NER model-based entity extraction
    - MyGene.info API for synonym expansion
    - LRU cache for gene aliases
    """

    def __init__(self, ner_pipeline: Optional[Any] = None):
        """
        Initialize Entity Analyzer.

        Args:
            ner_pipeline: Biomedical NER pipeline (d4data/biomedical-ner-all)
        """
        self.ner_pipeline = ner_pipeline
        self.gene_alias_cache = LRUCache(settings.GENE_ALIAS_CACHE_SIZE)

        logger.info(
            "EntityAnalyzer initialized",
            use_ner=ner_pipeline is not None,
            cache_size=settings.GENE_ALIAS_CACHE_SIZE
        )

    async def extract(self, query: str) -> Dict[str, Set[str]]:
        """
        Extract entities from query.

        Args:
            query: Input query string

        Returns:
            Dictionary with keys: 'genes', 'drugs', 'keywords'
        """
        entities = {
            "genes": set(),
            "drugs": set(),
            "keywords": set()
        }

        # Step 1: Rule-based gene extraction
        genes = self._extract_genes_regex(query)
        entities["genes"].update(genes)

        # Step 2: NER-based extraction
        if self.ner_pipeline:
            ner_entities = self._extract_ner_entities(query)
            entities["genes"].update(ner_entities.get("genes", set()))
            entities["drugs"].update(ner_entities.get("drugs", set()))

        # Step 3: Hardcoded biomedical keywords (matching original behavior).
        hardcoded_keywords = [
            "synthetic lethal",
            "synthetic lethality",
            "collateral lethality",
            "pathway",
            "inhibitor",
            "mutation",
            "expression",
            "screening",
            "crispr",
            "rnai",
            "knockdown",
            "cancer",
            "tumor",
            "therapy",
            "drug",
            "mechanism",
            "合成致死",
            "缺失",
            "抑制剂",
        ]
        query_lower = query.lower()
        for keyword in hardcoded_keywords:
            if keyword.lower() in query_lower:
                entities["keywords"].add(keyword)

        logger.debug(
            "Entity extraction completed",
            num_genes=len(entities["genes"]),
            num_drugs=len(entities["drugs"])
        )

        return entities

    def _extract_genes_regex(self, text: str) -> Set[str]:
        """
        Rule-based gene name extraction.

        Pattern: Uppercase letter + alphanumeric (2-10 chars)
        Handles Chinese adjacency (e.g., "STK11缺失")

        Args:
            text: Input text

        Returns:
            Set of potential gene symbols
        """
        # Pattern: non-word-boundary to handle Chinese
        gene_pattern = r'(?:^|[^A-Z0-9])([A-Z][A-Z0-9]{1,9})(?=$|[^A-Z0-9])'
        potential_genes = re.findall(gene_pattern, text.upper())

        # Filter blacklist
        blacklist = {"AND", "NOT", "THE", "FOR", "WITH", "FROM", "OR"}
        genes = {
            g for g in potential_genes
            if len(g) >= 2 and g not in blacklist
        }

        return genes

    def _extract_ner_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        NER model-based entity extraction.

        Args:
            text: Input text

        Returns:
            Dictionary with 'genes' and 'drugs' sets
        """
        entities = {"genes": set(), "drugs": set()}

        try:
            ner_results = self.ner_pipeline(text)

            # Entity type mapping
            gene_like_labels = {
                'Gene_or_gene_product',
                'Gene',
                'Protein',
                'Coreference',
                'Diagnostic_procedure'
            }

            drug_like_labels = {
                'Drug',
                'Chemical',
                'Medication',
                'Pharmacologic_substance'
            }

            for entity in ner_results:
                word = entity['word'].strip().replace('##', '')  # Clean BERT tokens
                group = entity.get('entity_group', '')

                if len(word) > 1:
                    if group in gene_like_labels:
                        entities["genes"].add(word.upper())
                    elif group in drug_like_labels:
                        entities["drugs"].add(word)

        except Exception as e:
            logger.warning("NER extraction failed", error=str(e))

        return entities

    async def expand_gene_aliases(
        self,
        genes: List[str],
        max_aliases: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Batch query MyGene.info for gene aliases.

        Args:
            genes: List of gene symbols
            max_aliases: Maximum aliases per gene (default: from settings)

        Returns:
            Dictionary mapping gene -> list of aliases (including official symbol)
        """
        if not genes or not settings.USE_GENE_SYNONYM_EXPANSION:
            return {}

        max_aliases = max_aliases or settings.MYGENE_MAX_ALIASES

        # Filter already cached genes
        genes_to_query = [g for g in genes if g not in self.gene_alias_cache]

        if not genes_to_query:
            # All cached
            return {g: self.gene_alias_cache.get(g, [g]) for g in genes}

        # Query MyGene.info
        result_map = {}

        try:
            url = "https://mygene.info/v3/query"
            payload = {
                "q": genes_to_query,
                "scopes": "symbol,alias",
                "fields": "symbol,alias,ensembl.gene",
                "species": "human"
            }

            # Retry with exponential backoff
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers={"User-Agent": "SLAgent/1.0"}
                        )
                        data = response.json()

                        # Parse results
                        for item in data:
                            if isinstance(item, dict):
                                query_gene = item.get('query', '')
                                official_symbol = item.get('symbol', query_gene)
                                aliases_raw = item.get('alias', [])

                                # Normalize alias list
                                if isinstance(aliases_raw, list):
                                    aliases = [official_symbol] + aliases_raw[:max_aliases]
                                elif isinstance(aliases_raw, str):
                                    aliases = [official_symbol, aliases_raw]
                                else:
                                    aliases = [official_symbol]

                                # Deduplicate and filter short aliases
                                aliases = list(OrderedDict.fromkeys(
                                    [a for a in aliases if len(a) > 1]
                                ))

                                result_map[query_gene] = aliases

                                # Update cache
                                if len(self.gene_alias_cache) < settings.GENE_ALIAS_CACHE_SIZE:
                                    self.gene_alias_cache[query_gene] = aliases

                        logger.debug(
                            "Gene aliases fetched",
                            num_genes=len(genes_to_query),
                            total_aliases=sum(len(v) for v in result_map.values())
                        )
                        break  # Success

                except Exception as e:
                    wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                    logger.warning(
                        "MyGene.info request failed",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e)
                    )
                    if attempt < 2:
                        await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error("Gene alias expansion failed", error=str(e))

        # Merge with cache
        for gene in genes:
            if gene not in result_map:
                result_map[gene] = self.gene_alias_cache.get(gene, [gene])

        return result_map

    def build_expanded_query(
        self,
        query: str,
        gene_alias_map: Dict[str, List[str]]
    ) -> str:
        """
        Build Boolean query with gene aliases.

        Example:
            (TP53 OR P53 OR TRP53) AND ("synthetic lethality" OR "synthetic lethal")

        Args:
            query: Original query
            gene_alias_map: Gene -> aliases mapping

        Returns:
            Expanded Boolean query
        """
        all_aliases = []
        for gene, aliases in gene_alias_map.items():
            all_aliases.extend(aliases[:3])  # Max 3 per gene

        if not all_aliases:
            return query

        # Build OR clause for genes
        alias_group = " OR ".join([f'"{a}"' for a in all_aliases])

        # Add SL context
        expanded_query = f'({alias_group}) AND ("synthetic lethality" OR "synthetic lethal")'

        logger.debug("Expanded query built", length=len(expanded_query))

        return expanded_query
