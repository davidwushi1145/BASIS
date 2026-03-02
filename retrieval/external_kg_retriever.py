"""
External Knowledge Graph Retriever

SQLite-based retrieval for external knowledge graph data.
Features:
- High-performance SQLite queries
- Bidirectional relationship traversal (x→y and y←x)
- Priority relation sorting
- 1-hop and 2-hop subgraph extraction
"""

import sqlite3
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from retrieval.base import BaseRetriever, RetrievalResult, SourceType
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ExternalKGRetriever(BaseRetriever):
    """
    High-performance SQLite-based Knowledge Graph Retriever.

    Based on kg.db with 12M+ triples from DepMap, DrugBank, Bgee.

    Features:
    - Zero load time - instant connect
    - Low memory usage (<10MB)
    - Millisecond queries via B-Tree indexes
    - Bidirectional queries (x→y and y←x)

    Confidence: P0 (Highest) - Structured data, more reliable than text retrieval.
    """

    def __init__(
        self,
        db_path: str = None,
        source_type: SourceType = SourceType.EXTERNAL_KG
    ):
        """
        Initialize External KG Retriever.

        Args:
            db_path: Path to SQLite database
            source_type: Source type for results
        """
        super().__init__(source_type)
        self.db_path = db_path or getattr(settings, 'EXTERNAL_KG_DB_PATH', None)
        self.available = False
        self.stats = {'total_rows': 0, 'unique_relations': 0}

        if not settings.USE_EXTERNAL_KG:
            logger.info("External KG disabled by config")
            return

        if not self.db_path or not Path(self.db_path).exists():
            logger.warning("External KG database not found", path=self.db_path)
            return

        try:
            # Test connection and get stats
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM relationships')
            self.stats['total_rows'] = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT relation) FROM relationships')
            self.stats['unique_relations'] = cursor.fetchone()[0]

            conn.close()

            self.available = True
            logger.info(
                "External KG connected",
                db_path=self.db_path,
                total_rows=self.stats['total_rows'],
                unique_relations=self.stats['unique_relations']
            )

        except Exception as e:
            logger.error("Failed to connect to External KG", error=str(e))

    def _get_connection(self):
        """Get thread-safe database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query_subgraph(
        self,
        entities: List[str],
        hops: int = None
    ) -> str:
        """
        Query entity relationships with 1-hop or 2-hop traversal.

        This is the core query method matching the original implementation.

        Args:
            entities: Gene or drug names (e.g., ['MTAP', 'PRMT5', 'Olaparib'])
            hops: Traversal depth (None = use config default)

        Returns:
            Formatted knowledge graph facts for LLM context
        """
        if not self.available or not entities:
            return ""

        if hops is None:
            hops = settings.EXTERNAL_KG_HOPS

        # Normalize entity names
        entities = [e.strip().upper() for e in entities if e]
        if not entities:
            return ""

        logger.debug("External KG querying entities", count=len(entities), entities=entities[:5])

        conn = self._get_connection()
        cursor = conn.cursor()

        found_paths: Set[str] = set()
        priority_paths: Set[str] = set()

        # Priority relations (uppercase for comparison)
        priority_rels_upper = {rel.upper() for rel in settings.EXTERNAL_KG_PRIORITY_RELATIONS}

        try:
            for entity in entities:
                # === Forward query: x → relation → y ===
                cursor.execute("""
                    SELECT x_name, relation, display_relation, y_name, y_type, rel_source
                    FROM relationships
                    WHERE x_name = ?
                    LIMIT 50
                """, (entity,))

                for row in cursor.fetchall():
                    path = f"- {row['x_name']} → {row['display_relation']} → {row['y_name']} (Type: {row['y_type']}, Source: {row['rel_source']})"

                    if row['relation'].upper() in priority_rels_upper:
                        priority_paths.add(path)
                    else:
                        found_paths.add(path)

                    # === 2-Hop query: x → y → z ===
                    if hops > 1:
                        target_node = row['y_name']
                        cursor.execute("""
                            SELECT relation, display_relation, y_name, y_type
                            FROM relationships
                            WHERE x_name = ?
                            LIMIT 10
                        """, (target_node,))

                        for sub in cursor.fetchall():
                            # Pruning: prevent returning to origin
                            if sub['y_name'] == entity:
                                continue

                            path_2hop = f"- {row['x_name']} → {row['display_relation']} → {target_node} → {sub['display_relation']} → {sub['y_name']}"

                            if sub['relation'].upper() in priority_rels_upper:
                                priority_paths.add(path_2hop)
                            else:
                                found_paths.add(path_2hop)

                # === Reverse query: y ← relation ← x (very important!) ===
                # e.g., Which CellLines depend on this Gene
                cursor.execute("""
                    SELECT x_name, x_type, display_relation, y_name, rel_source, relation
                    FROM relationships
                    WHERE y_name = ?
                    LIMIT 50
                """, (entity,))

                for row in cursor.fetchall():
                    path = f"- {row['x_name']} ({row['x_type']}) → {row['display_relation']} → {row['y_name']} (Source: {row['rel_source']})"

                    if row['relation'].upper() in priority_rels_upper:
                        priority_paths.add(path)
                    else:
                        found_paths.add(path)

        except Exception as e:
            logger.error("External KG query error", error=str(e))
        finally:
            conn.close()

        # Combine results: priority paths first
        all_paths = list(priority_paths) + list(found_paths)

        if not all_paths:
            logger.debug("No External KG paths found", entities=entities)
            return ""

        # Limit returned paths
        limited_paths = all_paths[:settings.EXTERNAL_KG_MAX_PATHS]

        # Format as natural language
        result = f"=== [External Knowledge Graph - {len(limited_paths)} Facts] ===\n"
        result += "Data Source: DepMap, DrugBank, Bgee (Anatomical Expression)\n"
        result += "Confidence Level: P0 (Highest - Structured Data)\n"
        result += f"Query: {', '.join(entities)}\n\n"
        result += "\n".join(limited_paths)

        logger.info(
            "External KG query complete",
            total_paths=len(limited_paths),
            priority=len(priority_paths),
            standard=len(found_paths)
        )

        return result

    async def search(
        self,
        query: str,
        top_k: int = 10,
        genes: List[str] = None,
        drugs: List[str] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search external knowledge graph.

        Args:
            query: Search query (used for context)
            top_k: Maximum results
            genes: List of gene names to search
            drugs: List of drug names to search

        Returns:
            List of RetrievalResult
        """
        if not self.available:
            return []

        # Combine entities
        entities = []
        if genes:
            entities.extend(genes)
        if drugs:
            entities.extend(drugs)

        # If no entities provided, try to extract from query
        if not entities:
            import re
            potential = re.findall(r'\b[A-Z][A-Z0-9]{1,9}\b', query)
            entities = potential[:5]

        if not entities:
            return []

        # Use core query method
        content = self.query_subgraph(entities, hops=settings.EXTERNAL_KG_HOPS)

        if not content:
            return []

        # Return as single comprehensive result
        return [RetrievalResult(
            content=content,
            score=0.95,  # P0 priority = high score
            source=self.source_type,
            metadata={
                'entities': entities,
                'type': 'external_kg',
                'priority': 'P0'
            }
        )]

    async def _search_genes(self, genes: List[str], top_k: int) -> List[RetrievalResult]:
        """Search for gene-related knowledge."""
        results = []
        if not self.available:
            return results

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            for gene in genes[:5]:  # Limit genes to prevent too many queries
                # Search synthetic lethality relationships
                cursor.execute("""
                    SELECT gene_a, gene_b, sl_score, evidence_type, source, pmid
                    FROM synthetic_lethality
                    WHERE gene_a = ? OR gene_b = ?
                    ORDER BY sl_score DESC
                    LIMIT ?
                """, (gene.upper(), gene.upper(), top_k))

                for row in cursor.fetchall():
                    content = self._format_sl_result(dict(row))
                    results.append(RetrievalResult(
                        content=content,
                        score=float(row['sl_score']) if row['sl_score'] else 0.8,
                        source=self.source_type,
                        metadata={
                            'gene_a': row['gene_a'],
                            'gene_b': row['gene_b'],
                            'evidence_type': row['evidence_type'],
                            'pmid': row['pmid']
                        }
                    ))

                # Search gene info
                cursor.execute("""
                    SELECT symbol, name, function, pathway, essentiality_score
                    FROM genes
                    WHERE symbol = ?
                """, (gene.upper(),))

                row = cursor.fetchone()
                if row:
                    content = self._format_gene_info(dict(row))
                    results.append(RetrievalResult(
                        content=content,
                        score=0.9,
                        source=self.source_type,
                        metadata={'gene': row['symbol'], 'type': 'gene_info'}
                    ))

        except sqlite3.Error as e:
            logger.error("Gene search failed", error=str(e))
        finally:
            conn.close()

        return results

    async def _search_drugs(self, drugs: List[str], top_k: int) -> List[RetrievalResult]:
        """Search for drug-related knowledge."""
        results = []
        if not self.available:
            return results

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            for drug in drugs[:5]:
                # Search drug-gene interactions
                cursor.execute("""
                    SELECT drug_name, target_gene, interaction_type,
                           mechanism, clinical_status, source
                    FROM drug_gene_interactions
                    WHERE LOWER(drug_name) LIKE ?
                    LIMIT ?
                """, (f'%{drug.lower()}%', top_k))

                for row in cursor.fetchall():
                    content = self._format_drug_result(dict(row))
                    results.append(RetrievalResult(
                        content=content,
                        score=0.85,
                        source=self.source_type,
                        metadata={
                            'drug': row['drug_name'],
                            'target': row['target_gene'],
                            'type': 'drug_gene'
                        }
                    ))

        except sqlite3.Error as e:
            logger.error("Drug search failed", error=str(e))
        finally:
            conn.close()

        return results

    async def _search_text(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Full-text search in knowledge graph."""
        results = []
        if not self.available:
            return results

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Search in FTS table if available
            cursor.execute("""
                SELECT content, score, metadata
                FROM kg_fts
                WHERE kg_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k))

            for row in cursor.fetchall():
                results.append(RetrievalResult(
                    content=row['content'],
                    score=0.7,
                    source=self.source_type,
                    metadata={'type': 'fts_match'}
                ))

        except sqlite3.Error:
            # FTS table may not exist
            pass
        finally:
            conn.close()

        return results

    def _format_sl_result(self, row: Dict) -> str:
        """Format synthetic lethality result."""
        lines = [
            f"Synthetic Lethality: {row['gene_a']} ↔ {row['gene_b']}",
            f"SL Score: {row.get('sl_score', 'N/A')}",
            f"Evidence: {row.get('evidence_type', 'N/A')}",
        ]
        if row.get('pmid'):
            lines.append(f"PMID: {row['pmid']}")
        lines.append(f"[Source: {row.get('source', 'External KG')}]")
        return '\n'.join(lines)

    def _format_gene_info(self, row: Dict) -> str:
        """Format gene information."""
        lines = [
            f"Gene: {row['symbol']} ({row.get('name', '')})",
            f"Function: {row.get('function', 'N/A')}",
        ]
        if row.get('pathway'):
            lines.append(f"Pathway: {row['pathway']}")
        if row.get('essentiality_score'):
            lines.append(f"Essentiality Score: {row['essentiality_score']}")
        lines.append("[Source: External KG]")
        return '\n'.join(lines)

    def _format_drug_result(self, row: Dict) -> str:
        """Format drug-gene interaction result."""
        lines = [
            f"Drug-Gene Interaction: {row['drug_name']} → {row['target_gene']}",
            f"Type: {row.get('interaction_type', 'N/A')}",
        ]
        if row.get('mechanism'):
            lines.append(f"Mechanism: {row['mechanism']}")
        if row.get('clinical_status'):
            lines.append(f"Clinical Status: {row['clinical_status']}")
        lines.append(f"[Source: {row.get('source', 'External KG')}]")
        return '\n'.join(lines)

    async def get_sl_partners(self, gene: str) -> List[Dict[str, Any]]:
        """
        Get all synthetic lethal partners for a gene.

        Args:
            gene: Gene symbol

        Returns:
            List of SL partner information
        """
        if not self.available:
            return []

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    CASE WHEN gene_a = ? THEN gene_b ELSE gene_a END as partner,
                    sl_score, evidence_type, source
                FROM synthetic_lethality
                WHERE gene_a = ? OR gene_b = ?
                ORDER BY sl_score DESC
            """, (gene.upper(), gene.upper(), gene.upper()))

            return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            logger.error("Get SL partners failed", gene=gene, error=str(e))
            return []
        finally:
            conn.close()

    async def get_drug_targets(self, drug: str) -> List[Dict[str, Any]]:
        """
        Get all targets for a drug.

        Args:
            drug: Drug name

        Returns:
            List of target information
        """
        if not self.available:
            return []

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT target_gene, interaction_type, mechanism
                FROM drug_gene_interactions
                WHERE LOWER(drug_name) LIKE ?
            """, (f'%{drug.lower()}%',))

            return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            logger.error("Get drug targets failed", drug=drug, error=str(e))
            return []
        finally:
            conn.close()

    def is_available(self) -> bool:
        """Check if retriever is available."""
        return self.available

    def close(self):
        """Close database connection (no-op for per-request connections)."""
        pass

    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """
        Get detailed info for a single node.

        Args:
            node_name: Node name to query

        Returns:
            Dict with exists, type, out_degree, in_degree, total_relations
        """
        if not self.available:
            return {'exists': False}

        node_upper = node_name.strip().upper()
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Count outgoing edges (x → y)
            cursor.execute('SELECT COUNT(*) FROM relationships WHERE x_name = ?', (node_upper,))
            out_degree = cursor.fetchone()[0]

            # Count incoming edges (y ← x)
            cursor.execute('SELECT COUNT(*) FROM relationships WHERE y_name = ?', (node_upper,))
            in_degree = cursor.fetchone()[0]

            if out_degree == 0 and in_degree == 0:
                return {'exists': False}

            # Get node type
            cursor.execute('SELECT x_type FROM relationships WHERE x_name = ? LIMIT 1', (node_upper,))
            row = cursor.fetchone()
            node_type = row['x_type'] if row else 'Unknown'

            return {
                'exists': True,
                'type': node_type,
                'out_degree': out_degree,
                'in_degree': in_degree,
                'total_relations': out_degree + in_degree
            }

        except Exception as e:
            logger.error("get_node_info error", error=str(e))
            return {'exists': False}
        finally:
            conn.close()
