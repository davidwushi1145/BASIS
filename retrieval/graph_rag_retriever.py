"""
Graph RAG Retriever

Dynamic knowledge graph construction and retrieval.
Features:
- LLM-based triplet extraction from retrieved text
- MyGene.info entity normalization
- Open Targets validation
- Multi-hop graph traversal
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import json

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from retrieval.base import BaseRetriever, RetrievalResult, SourceType
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Triplet Extraction Prompt (pipe-separated format)
TRIPLET_EXTRACTION_PROMPT = """Extract biomedical relationships from the text.
Format strictly as: Entity1|Relation|Entity2
Entities should be Genes, Drugs, Diseases, or Mechanisms.
Use SHORT phrases. Max 5 triplets.

Example:
MTAP|is synthetic lethal with|PRMT5
KRAS mutation|causes|drug resistance
PARP1|is inhibited by|Olaparib

Text: {text}

Output only the triplets, one per line. No headers or explanations."""

# Common gene alias mappings
GENE_ALIAS_MAP = {
    "p53": "TP53", "P53": "TP53",
    "her2": "ERBB2", "HER2": "ERBB2", "neu": "ERBB2",
    "ras": "KRAS", "k-ras": "KRAS",
    "arf": "CDKN2A", "p14arf": "CDKN2A", "p16": "CDKN2A",
    "myc": "MYC", "c-myc": "MYC",
    "egfr": "EGFR", "her1": "EGFR",
}


class GraphRAGRetriever(BaseRetriever):
    """
    Dynamic Graph RAG with LLM triplet extraction.

    Features:
    - LLM-based triplet extraction from text chunks
    - MyGene.info entity normalization
    - Open Targets relationship validation
    - Multi-hop graph traversal
    - Subgraph extraction and path finding
    """

    def __init__(
        self,
        graph_path: str = None,
        llm_client: Any = None,
        model_name: str = None,
        source_type: SourceType = SourceType.GRAPH_RAG
    ):
        """
        Initialize Graph RAG Retriever.

        Args:
            graph_path: Path to serialized graph (JSON/GraphML)
            llm_client: AsyncOpenAI client for triplet extraction
            model_name: Model name for triplet extraction
            source_type: Source type for results
        """
        super().__init__(source_type)

        if nx is None:
            logger.error("NetworkX not installed, Graph RAG unavailable")
            self._graph = None
            return

        self.graph_path = graph_path or getattr(settings, 'GRAPH_RAG_PATH', None)
        self._graph: Optional[nx.DiGraph] = None
        self.llm_client = llm_client
        self.model_name = model_name or settings.SOLVER_MODEL_NAME

        # Caches for normalization
        self.normalization_cache: Dict[str, str] = {}
        self.ensembl_id_cache: Dict[str, str] = {}
        self.efo_cache: Dict[str, str] = {}
        self.ot_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        if self.graph_path:
            self._load_graph()
        else:
            # Initialize empty graph
            self._graph = nx.DiGraph()
            logger.info("Empty Graph RAG initialized")

    def set_llm_client(self, client: Any, model_name: str = None):
        """Set LLM client for triplet extraction."""
        self.llm_client = client
        if model_name:
            self.model_name = model_name
        logger.info("LLM client set for Graph RAG", model=self.model_name)

    def _load_graph(self):
        """Load graph from file."""
        path = Path(self.graph_path)

        if not path.exists():
            logger.warning("Graph file not found", path=str(path))
            self._graph = nx.DiGraph()
            return

        try:
            if path.suffix == '.json':
                self._load_json_graph(path)
            elif path.suffix in ('.graphml', '.gml'):
                self._graph = nx.read_graphml(str(path))
            else:
                logger.warning("Unknown graph format", suffix=path.suffix)
                self._graph = nx.DiGraph()
                return

            logger.info(
                "Graph loaded",
                nodes=self._graph.number_of_nodes(),
                edges=self._graph.number_of_edges()
            )

        except Exception as e:
            logger.error("Failed to load graph", error=str(e))
            self._graph = nx.DiGraph()

    def _load_json_graph(self, path: Path):
        """Load graph from JSON format."""
        with open(path, 'r') as f:
            data = json.load(f)

        self._graph = nx.DiGraph()

        # Load nodes
        for node in data.get('nodes', []):
            node_id = node.get('id')
            attrs = {k: v for k, v in node.items() if k != 'id'}
            self._graph.add_node(node_id, **attrs)

        # Load edges
        for edge in data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            attrs = {k: v for k, v in edge.items() if k not in ('source', 'target')}
            self._graph.add_edge(source, target, **attrs)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        genes: List[str] = None,
        max_hops: int = 2,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search graph for relevant subgraphs.

        Args:
            query: Search query
            top_k: Maximum results
            genes: Seed genes for traversal
            max_hops: Maximum hops from seed nodes

        Returns:
            List of RetrievalResult
        """
        if not self._graph or self._graph.number_of_nodes() == 0:
            return []

        results = []

        # Find seed nodes
        seed_nodes = self._find_seed_nodes(genes or [], query)

        if not seed_nodes:
            logger.debug("No seed nodes found for query", query=query[:50])
            return []

        # Extract subgraph around seeds
        subgraph_nodes = self._expand_subgraph(seed_nodes, max_hops)

        # Extract relevant paths and relationships
        for node in subgraph_nodes:
            if len(results) >= top_k:
                break

            content = self._format_node_context(node, max_hops=1)
            if content:
                node_data = self._graph.nodes.get(node, {})
                results.append(RetrievalResult(
                    content=content,
                    score=self._compute_node_relevance(node, seed_nodes),
                    source=self.source_type,
                    metadata={
                        'node': node,
                        'type': node_data.get('type', 'unknown'),
                        'hops_from_seed': self._min_distance_to_seeds(node, seed_nodes)
                    }
                ))

        # Sort by relevance
        results.sort(key=lambda x: x.score, reverse=True)

        logger.debug("Graph search completed", seeds=len(seed_nodes), results=len(results))
        return results[:top_k]

    def _find_seed_nodes(self, genes: List[str], query: str) -> Set[str]:
        """Find seed nodes from genes and query."""
        seeds = set()

        # Add gene nodes
        for gene in genes:
            gene_upper = gene.upper()
            if self._graph.has_node(gene_upper):
                seeds.add(gene_upper)
            # Try lowercase
            elif self._graph.has_node(gene.lower()):
                seeds.add(gene.lower())

        # If no genes found, try to find nodes matching query terms
        if not seeds:
            query_terms = set(query.upper().split())
            for node in self._graph.nodes():
                node_upper = str(node).upper()
                if node_upper in query_terms or any(t in node_upper for t in query_terms):
                    seeds.add(node)
                    if len(seeds) >= 5:  # Limit seeds from query matching
                        break

        return seeds

    def _expand_subgraph(self, seed_nodes: Set[str], max_hops: int) -> Set[str]:
        """Expand subgraph from seed nodes."""
        expanded = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(max_hops):
            new_frontier = set()
            for node in frontier:
                # Add successors
                new_frontier.update(self._graph.successors(node))
                # Add predecessors
                new_frontier.update(self._graph.predecessors(node))

            new_frontier -= expanded  # Only new nodes
            expanded.update(new_frontier)
            frontier = new_frontier

            if not frontier:
                break

        return expanded

    def _format_node_context(self, node: str, max_hops: int = 1) -> str:
        """Format node and its immediate context as text."""
        if not self._graph.has_node(node):
            return ""

        node_data = self._graph.nodes[node]
        lines = []

        # Node info
        node_type = node_data.get('type', 'Entity')
        node_name = node_data.get('name', node)
        lines.append(f"{node_type}: {node_name}")

        # Node attributes
        for key, value in node_data.items():
            if key not in ('type', 'name', 'id') and value:
                lines.append(f"  {key}: {value}")

        # Outgoing relationships
        out_edges = list(self._graph.out_edges(node, data=True))
        if out_edges:
            lines.append("")
            lines.append("Relationships:")
            for _, target, edge_data in out_edges[:10]:  # Limit edges
                rel_type = edge_data.get('relationship', edge_data.get('type', 'related_to'))
                target_name = self._graph.nodes.get(target, {}).get('name', target)
                lines.append(f"  → {rel_type} → {target_name}")

        # Incoming relationships
        in_edges = list(self._graph.in_edges(node, data=True))
        if in_edges:
            if not out_edges:
                lines.append("")
                lines.append("Relationships:")
            for source, _, edge_data in in_edges[:10]:
                rel_type = edge_data.get('relationship', edge_data.get('type', 'related_to'))
                source_name = self._graph.nodes.get(source, {}).get('name', source)
                lines.append(f"  ← {rel_type} ← {source_name}")

        lines.append("[Source: Knowledge Graph]")

        return '\n'.join(lines)

    def _compute_node_relevance(self, node: str, seeds: Set[str]) -> float:
        """Compute relevance score for a node."""
        if node in seeds:
            return 1.0

        # Score based on connectivity to seeds
        connections = 0
        for seed in seeds:
            if self._graph.has_edge(seed, node) or self._graph.has_edge(node, seed):
                connections += 1

        # Also consider node degree
        degree = self._graph.degree(node)

        # Combine factors
        return min(0.9, 0.3 + (connections * 0.2) + (min(degree, 10) * 0.02))

    def _min_distance_to_seeds(self, node: str, seeds: Set[str]) -> int:
        """Get minimum distance to any seed node."""
        if node in seeds:
            return 0

        min_dist = float('inf')
        for seed in seeds:
            try:
                dist = nx.shortest_path_length(self._graph, seed, node)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                pass
            try:
                dist = nx.shortest_path_length(self._graph, node, seed)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                pass

        return min_dist if min_dist != float('inf') else -1

    async def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 4
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find paths between two entities.

        Args:
            source: Source node
            target: Target node
            max_length: Maximum path length

        Returns:
            List of paths, where each path is a list of (node, edge_type, node) tuples
        """
        if not self._graph:
            return []

        # Normalize node names
        source = source.upper() if self._graph.has_node(source.upper()) else source
        target = target.upper() if self._graph.has_node(target.upper()) else target

        if not self._graph.has_node(source) or not self._graph.has_node(target):
            return []

        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self._graph, source, target,
                cutoff=max_length
            ))

            # Format paths with edge info
            formatted_paths = []
            for path in paths[:10]:  # Limit number of paths
                formatted = []
                for i in range(len(path) - 1):
                    edge_data = self._graph.get_edge_data(path[i], path[i + 1]) or {}
                    rel_type = edge_data.get('relationship', edge_data.get('type', 'related'))
                    formatted.append((path[i], rel_type, path[i + 1]))
                formatted_paths.append(formatted)

            return formatted_paths

        except nx.NetworkXNoPath:
            return []

    async def get_neighbors(
        self,
        node: str,
        relationship_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node.

        Args:
            node: Node identifier
            relationship_type: Filter by relationship type

        Returns:
            List of neighbor info
        """
        if not self._graph or not self._graph.has_node(node):
            return []

        neighbors = []

        # Outgoing edges
        for _, target, edge_data in self._graph.out_edges(node, data=True):
            rel_type = edge_data.get('relationship', edge_data.get('type', ''))
            if relationship_type is None or rel_type == relationship_type:
                neighbors.append({
                    'node': target,
                    'direction': 'outgoing',
                    'relationship': rel_type,
                    'data': self._graph.nodes.get(target, {})
                })

        # Incoming edges
        for source, _, edge_data in self._graph.in_edges(node, data=True):
            rel_type = edge_data.get('relationship', edge_data.get('type', ''))
            if relationship_type is None or rel_type == relationship_type:
                neighbors.append({
                    'node': source,
                    'direction': 'incoming',
                    'relationship': rel_type,
                    'data': self._graph.nodes.get(source, {})
                })

        return neighbors

    def add_node(self, node_id: str, **attributes):
        """Add a node to the graph."""
        if self._graph:
            self._graph.add_node(node_id, **attributes)

    def add_edge(self, source: str, target: str, **attributes):
        """Add an edge to the graph."""
        if self._graph:
            self._graph.add_edge(source, target, **attributes)

    def save_graph(self, path: str = None):
        """Save graph to file."""
        save_path = Path(path or self.graph_path)
        if not save_path or not self._graph:
            return

        try:
            if save_path.suffix == '.json':
                data = {
                    'nodes': [
                        {'id': n, **d}
                        for n, d in self._graph.nodes(data=True)
                    ],
                    'edges': [
                        {'source': u, 'target': v, **d}
                        for u, v, d in self._graph.edges(data=True)
                    ]
                }
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                nx.write_graphml(self._graph, str(save_path))

            logger.info("Graph saved", path=str(save_path))

        except Exception as e:
            logger.error("Failed to save graph", error=str(e))

    def is_available(self) -> bool:
        """Check if retriever is available."""
        return self._graph is not None and nx is not None

    # ==================== LLM Triplet Extraction ====================

    async def extract_triplets(
        self,
        text_chunks: List[Dict[str, Any]],
        max_chunks: int = None
    ) -> List[Tuple[str, str, str]]:
        """
        Use LLM to extract triplets from text chunks concurrently.

        Args:
            text_chunks: List of dicts with 'content' key
            max_chunks: Maximum chunks to process

        Returns:
            List of (subject, relation, object) tuples
        """
        if not self.llm_client:
            logger.warning("No LLM client for triplet extraction")
            return []

        max_chunks = max_chunks or settings.MAX_TRIPLET_CHUNKS
        target_chunks = text_chunks[:max_chunks]

        if not target_chunks:
            logger.debug("No chunks for triplet extraction")
            return []

        logger.info("Extracting triplets", chunks=len(target_chunks))

        async def _extract_single(text: str) -> List[Tuple[str, str, str]]:
            """Extract triplets from single chunk."""
            try:
                if len(text) < 50:
                    return []

                # Truncate input to prevent excess
                truncated = text[:1500]

                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{
                            "role": "user",
                            "content": TRIPLET_EXTRACTION_PROMPT.format(text=truncated)
                        }],
                        temperature=0.0,
                        max_tokens=getattr(settings, "TRIPLET_EXTRACTION_MAX_TOKENS", 512)
                    ),
                    timeout=getattr(settings, "TRIPLET_EXTRACTION_TIMEOUT_SECONDS", 65.0)
                )

                raw_output = response.choices[0].message.content
                triplets = []

                # Parse pipe-separated triplets
                for line in raw_output.strip().split('\n'):
                    line = line.strip()
                    if '|' in line and line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3 and all(parts):
                            triplets.append((parts[0], parts[1], parts[2]))

                logger.debug("Extracted triplets from chunk", count=len(triplets))
                return triplets

            except asyncio.TimeoutError:
                logger.warning("Triplet extraction timeout")
                return []
            except Exception as e:
                logger.error("Triplet extraction error", error=str(e))
                return []

        # Extract concurrently
        tasks = [_extract_single(chunk.get('content', '')) for chunk in target_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_triplets = []
        for res in results:
            if isinstance(res, list):
                all_triplets.extend(res)

        logger.info("Total triplets extracted", count=len(all_triplets))
        return all_triplets

    async def build_graph_from_chunks(
        self,
        text_chunks: List[Dict[str, Any]],
        normalize: bool = True
    ) -> int:
        """
        Build graph dynamically from text chunks.

        Args:
            text_chunks: Text chunks to extract from
            normalize: Whether to normalize entities

        Returns:
            Number of triplets added
        """
        if not settings.USE_GRAPH_RAG:
            return 0

        # Extract triplets
        triplets = await self.extract_triplets(text_chunks)

        if not triplets:
            return 0

        # Normalize entities if enabled
        if normalize and settings.USE_BIO_NORMALIZATION:
            entities = set()
            for s, _, o in triplets:
                entities.add(s)
                entities.add(o)

            norm_map = await self.normalize_entities(list(entities))
        else:
            norm_map = {}

        # Clear existing graph and rebuild
        self._graph = nx.DiGraph()

        # Add triplets to graph
        for subj, rel, obj in triplets:
            norm_subj = norm_map.get(subj, subj.upper())
            norm_obj = norm_map.get(obj, obj.upper())

            self._graph.add_node(norm_subj, type='Entity', name=norm_subj)
            self._graph.add_node(norm_obj, type='Entity', name=norm_obj)
            self._graph.add_edge(norm_subj, norm_obj, relationship=rel)

        logger.info(
            "Graph built from chunks",
            nodes=self._graph.number_of_nodes(),
            edges=self._graph.number_of_edges()
        )

        return len(triplets)

    # ==================== MyGene.info Normalization ====================

    async def normalize_entities(self, entities: List[str]) -> Dict[str, str]:
        """
        Batch normalize entities using MyGene.info.

        Args:
            entities: Entity names to normalize

        Returns:
            Mapping from original to normalized names
        """
        if not entities:
            return {}

        # First apply hardcoded mappings
        result = {}
        unknown = []

        for e in set(entities):
            e_lower = e.lower()
            if e_lower in GENE_ALIAS_MAP:
                result[e] = GENE_ALIAS_MAP[e_lower]
                self.normalization_cache[e] = GENE_ALIAS_MAP[e_lower]
            elif e in self.normalization_cache:
                result[e] = self.normalization_cache[e]
            else:
                unknown.append(e)

        # If normalization disabled or no httpx, use uppercase fallback
        if not settings.USE_BIO_NORMALIZATION or not HTTPX_AVAILABLE or not unknown:
            for e in unknown:
                result[e] = e.upper()
            return result

        logger.debug("MyGene batch normalizing", entities=len(unknown))

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://mygene.info/v3/query",
                    json={
                        "q": unknown,
                        "scopes": "symbol,alias,entrezgene,uniprot",
                        "fields": "symbol,ensembl.gene",
                        "species": "human",
                        "dotfield": False
                    },
                    timeout=settings.MYGENE_TIMEOUT
                )
                data = resp.json()

            # Parse results
            for item in data:
                if isinstance(item, dict):
                    original = item.get('query')
                    official = item.get('symbol')

                    # Extract Ensembl ID if available
                    ens_data = item.get('ensembl')
                    ensembl_id = None
                    if isinstance(ens_data, list) and ens_data:
                        ensembl_id = ens_data[0].get('gene')
                    elif isinstance(ens_data, dict):
                        ensembl_id = ens_data.get('gene')

                    if original:
                        normalized = official if official else original.upper()
                        result[original] = normalized
                        self.normalization_cache[original] = normalized

                        if ensembl_id:
                            self.ensembl_id_cache[normalized] = ensembl_id

            # Fallback for unmatched
            for e in unknown:
                if e not in result:
                    result[e] = e.upper()

            logger.debug("MyGene normalization complete", mapped=len(result))

        except Exception as e:
            logger.warning("MyGene normalization failed", error=str(e))
            for e in unknown:
                result[e] = e.upper()

        return result

    # ==================== STRING DB Validation ====================

    async def verify_with_string_db(
        self,
        gene_a: str,
        gene_b: str
    ) -> Optional[Dict[str, Any]]:
        """
        Verify gene-gene interaction using STRING DB API.

        Matching original verify_interaction_with_string implementation.

        Args:
            gene_a: First gene symbol
            gene_b: Second gene symbol

        Returns:
            Validation result with score if found
        """
        if not settings.USE_STRING_VALIDATION or not HTTPX_AVAILABLE:
            return None

        if not gene_a or not gene_b:
            return None

        # Build STRING API URL
        identifiers = f"{gene_a}%0d{gene_b}"
        url = f"https://string-db.org/api/json/network?identifiers={identifiers}&species=9606"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=2.0)
                data = resp.json()

            if data and len(data) > 0:
                score = data[0].get('score', 0)
                if score >= settings.STRING_DB_THRESHOLD:
                    result = {
                        'validated': True,
                        'score': score,
                        'source': 'STRING'
                    }
                    logger.debug(
                        "STRING DB validation positive",
                        gene_a=gene_a, gene_b=gene_b, score=score
                    )
                    return result

            return {'validated': False, 'score': 0, 'source': 'STRING'}

        except Exception as e:
            logger.warning("STRING DB validation failed", error=str(e))
            return None

    async def validate_triplet(
        self,
        subject: str,
        relation: str,
        obj: str
    ) -> Dict[str, Any]:
        """
        Validate a triplet using external APIs.

        Attempts:
        1. STRING DB for gene-gene interactions
        2. Open Targets for gene-disease associations

        Args:
            subject: Subject entity
            relation: Relationship type
            obj: Object entity

        Returns:
            Validation result
        """
        result = {
            'triplet': (subject, relation, obj),
            'validated': False,
            'source': None,
            'score': 0.0
        }

        # Normalize entities
        subj_upper = subject.upper()
        obj_upper = obj.upper()

        # Check if both look like genes (uppercase, 2-10 chars)
        is_gene_pattern = lambda s: len(s) >= 2 and len(s) <= 10 and s.isupper()

        if is_gene_pattern(subj_upper) and is_gene_pattern(obj_upper):
            # Try STRING DB validation for gene-gene
            string_result = await self.verify_with_string_db(subj_upper, obj_upper)
            if string_result and string_result.get('validated'):
                result.update(string_result)
                return result

        # Check if looks like gene-disease (subject is gene, object contains disease keywords)
        disease_keywords = settings.DISEASE_KEYWORDS if hasattr(settings, 'DISEASE_KEYWORDS') else {
            'CANCER', 'TUMOR', 'CARCINOMA', 'LEUKEMIA', 'LYMPHOMA'
        }

        is_disease = any(kw in obj_upper for kw in disease_keywords)

        if is_gene_pattern(subj_upper) and is_disease:
            # Try Open Targets validation for gene-disease
            ot_result = await self.validate_with_opentargets(subj_upper, obj)
            if ot_result and ot_result.get('validated'):
                result.update(ot_result)
                return result

        return result

    async def build_and_validate_graph(
        self,
        text_chunks: List[Dict[str, Any]],
        query_entities: Dict[str, Set[str]]
    ) -> Tuple[int, str]:
        """
        Build graph from chunks with validation.

        Matching original build_and_query_graph implementation.

        Args:
            text_chunks: Text chunks for triplet extraction
            query_entities: Extracted query entities (genes, keywords)

        Returns:
            Tuple of (num_triplets, formatted_context)
        """
        if not settings.USE_GRAPH_RAG:
            return 0, ""

        # Step 1: Extract triplets
        triplets = await self.extract_triplets(text_chunks)
        if not triplets:
            logger.debug("No triplets extracted")
            return 0, ""

        # Step 2: Normalize entities
        all_entities = set()
        for s, _, o in triplets:
            all_entities.add(s)
            all_entities.add(o)

        norm_map = await self.normalize_entities(list(all_entities))

        # Step 3: Build graph
        self._graph = nx.DiGraph() if nx else None
        if not self._graph:
            return 0, ""

        validated_count = 0
        ot_validations: List[str] = []
        for subj, rel, obj in triplets:
            norm_subj = norm_map.get(subj, subj.upper())
            norm_obj = norm_map.get(obj, obj.upper())

            # Add nodes
            self._graph.add_node(norm_subj, type='Entity', name=norm_subj)
            self._graph.add_node(norm_obj, type='Entity', name=norm_obj)

            # Validate if enabled
            validation = None
            if settings.USE_STRING_VALIDATION or settings.USE_OPENTARGETS_VALIDATION:
                validation = await self.validate_triplet(norm_subj, rel, norm_obj)
                if validation.get('validated'):
                    validated_count += 1
                    if str(validation.get('source', '')).lower() == "opentargets":
                        validation_text = validation.get("validation_text")
                        if validation_text:
                            ot_validations.append(validation_text)

            # Add edge with validation info
            edge_attrs = {'relationship': rel}
            if validation:
                edge_attrs['validated'] = validation.get('validated', False)
                edge_attrs['validation_score'] = validation.get('score', 0)
                edge_attrs['validation_source'] = validation.get('source', '')

            self._graph.add_edge(norm_subj, norm_obj, **edge_attrs)

        logger.info(
            "Graph built with validation",
            nodes=self._graph.number_of_nodes(),
            edges=self._graph.number_of_edges(),
            validated=validated_count
        )

        # Step 4: Format context
        genes = list(query_entities.get('genes', set()))
        keywords = list(query_entities.get('keywords', set()))[:3]
        seeds = genes + keywords
        context = self.format_graph_context(seeds)
        if context and ot_validations:
            deduped = list(dict.fromkeys(ot_validations))
            context += "\n\n[Gold Standard Validation (Open Targets)]\n" + "\n".join(
                f"- {item}" for item in deduped[:3]
            )

        return len(triplets), context

    async def validate_with_opentargets(
        self,
        gene: str,
        disease: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate gene-disease relationship using Open Targets.

        Args:
            gene: Gene symbol
            disease: Disease name

        Returns:
            Validation result with score if found
        """
        if not settings.USE_OPENTARGETS_VALIDATION or not HTTPX_AVAILABLE:
            return None

        cache_key = (gene.upper(), disease.upper())
        if cache_key in self.ot_cache:
            return self.ot_cache[cache_key]

        # Get Ensembl ID for gene
        ensembl_id = self.ensembl_id_cache.get(gene.upper())
        if not ensembl_id:
            # Try to fetch via MyGene
            norm_result = await self.normalize_entities([gene])
            normalized = norm_result.get(gene, gene.upper())
            ensembl_id = self.ensembl_id_cache.get(normalized)

        if not ensembl_id:
            return None

        # Get EFO ID for disease
        efo_id = await self._get_efo_id(disease)
        if not efo_id:
            return None

        try:
            query = """
            query TargetDiseaseAssociation($ensemblId: String!, $efoIds: [String!]!) {
              target(ensemblId: $ensemblId) {
                associatedDiseases(efoIds: $efoIds) {
                  rows {
                    score
                    datatypeScores {
                      id
                      score
                    }
                  }
                }
              }
            }
            """

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    settings.OPENTARGETS_GRAPHQL_URL,
                    json={
                        "query": query,
                        "variables": {
                            "ensemblId": ensembl_id,
                            "efoIds": [efo_id]
                        }
                    },
                    timeout=settings.OPENTARGETS_TIMEOUT
                )
                data = resp.json()

            # Parse result
            rows = data.get('data', {}).get('target', {}).get('associatedDiseases', {}).get('rows', [])
            if rows:
                score = float(rows[0].get('score', 0) or 0)
                datatype_scores = rows[0].get('datatypeScores', [])
                top_evidence = "N/A"
                if datatype_scores:
                    top = max(datatype_scores, key=lambda x: float(x.get('score', 0) or 0))
                    top_evidence = str(top.get('id', 'unknown')).replace("_", " ").title()

                credibility = "High" if score > 0.5 else "Moderate"
                result = {
                    'validated': score >= settings.OPENTARGETS_SCORE_THRESHOLD,
                    'score': score,
                    'source': 'OpenTargets',
                    'validation_text': (
                        f"OT Validation: {gene}-{disease} "
                        f"(Score: {score:.2f}, {credibility}, Top Evidence: {top_evidence})"
                    )
                }
                self.ot_cache[cache_key] = result
                return result

            result = {'validated': False, 'score': 0.0, 'source': 'OpenTargets'}
            self.ot_cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning("Open Targets validation failed", error=str(e))
            return None

    async def _get_efo_id(self, disease: str) -> Optional[str]:
        """Get EFO ID for a disease name."""
        disease_upper = disease.upper()

        if disease_upper in self.efo_cache:
            return self.efo_cache[disease_upper]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    settings.OPENTARGETS_SEARCH_URL,
                    params={"q": disease, "size": 5, "filter": "disease"},
                    timeout=settings.OPENTARGETS_TIMEOUT
                )
                data = resp.json()

            # Find best match. Open Targets may return either `hits` or `data`.
            hits = data.get('hits') or data.get('data', [])
            for hit in hits:
                if hit.get('entity') == 'disease':
                    efo_id = hit.get('id')
                    if efo_id:
                        self.efo_cache[disease_upper] = efo_id
                        return efo_id

            return None

        except Exception as e:
            logger.warning("EFO lookup failed", error=str(e))
            return None

    # ==================== Graph Context Formatting ====================

    def format_graph_context(
        self,
        genes: List[str],
        max_tokens: int = None
    ) -> str:
        """
        Format graph as context for LLM.

        Args:
            genes: Seed genes
            max_tokens: Maximum tokens for context

        Returns:
            Formatted graph context string
        """
        if not self._graph or self._graph.number_of_nodes() == 0:
            return ""

        lines = ["=== [Knowledge Graph Context] ==="]

        # Find seed nodes
        seeds = self._find_seed_nodes(genes, "")

        # Get relevant edges
        edges_found = []
        for seed in seeds:
            # Outgoing
            for _, target, data in self._graph.out_edges(seed, data=True):
                rel = data.get('relationship', 'related_to')
                # ev1-style triplet line format (pairs with _truncate_graph_context parsing).
                edges_found.append(f"{seed} --[{rel}]--> {target}")

            # Incoming
            for source, _, data in self._graph.in_edges(seed, data=True):
                rel = data.get('relationship', 'related_to')
                edges_found.append(f"{source} --[{rel}]--> {seed}")

        # Deduplicate while preserving order (more stable prompts).
        deduped_edges = []
        seen_edges = set()
        for edge in edges_found:
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            deduped_edges.append(edge)
        edges_found = deduped_edges[:20]

        if not edges_found:
            return ""

        lines.append(f"Query Genes: {', '.join(genes)}")
        lines.append(f"Extracted Relationships ({len(edges_found)}):")
        lines.extend(edges_found)
        lines.append("[Source: Dynamic Graph RAG]")

        result = '\n'.join(lines)
        return result
