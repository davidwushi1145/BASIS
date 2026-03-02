"""
Pydantic Settings for SLAgent

All configuration is managed through environment variables and this settings class.
"""

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # ==================== Model Paths ====================
    EMBEDDING_PATH: str = Field(
        ...,
        description="Path to Qwen3-Embedding-0.6B model"
    )
    RERANKER_PATH: str = Field(
        ...,
        description="Path to Qwen3-Reranker-0.6B model"
    )
    NER_MODEL_PATH: str = Field(
        ...,
        description="Path to biomedical NER model (d4data/biomedical-ner-all)"
    )

    # ==================== Corpus & Cache ====================
    CORPUS_FILE: str = Field(
        ...,
        description="Path to RAG corpus JSONL file"
    )
    VECTOR_CACHE: Optional[str] = Field(
        None,
        description="Path to precomputed vector cache (.pt file)"
    )

    # ==================== Device Configuration ====================
    DEVICE: str = Field(
        "cuda:0",
        description="Torch device (cuda:0, cuda:1, cpu)"
    )

    # ==================== API Keys ====================
    SERPER_API_KEY: str = Field(
        ...,
        description="Google Serper API key for web search"
    )
    API_KEY: str = Field(
        ...,
        description="OpenAI-compatible API key (intern-ai.org.cn)"
    )
    PROXY_URL: Optional[str] = Field(
        None,
        description="HTTP proxy URL (e.g., http://127.0.0.1:6152)"
    )
    SOLVER_API_BASE: str = Field(
        "https://chat.intern-ai.org.cn/api/v1",
        description="Base URL for Solver/Verifier API"
    )
    SOLVER_MODEL_NAME: str = Field(
        "intern-s1-mini",
        description="Model name for Solver"
    )
    VERIFIER_MODEL_NAME: str = Field(
        "intern-s1-mini",
        description="Model name for Verifier"
    )

    # NCBI/PubMed
    NCBI_EMAIL: str = Field(
        "your-email@example.com",
        description="Email for NCBI Entrez API (required by NCBI)"
    )
    NCBI_API_KEY: Optional[str] = Field(
        None,
        description="NCBI API key (increases rate limit to 10 req/s)"
    )
    NCBI_TOOL_NAME: str = Field(
        "SLAgent_RAG",
        description="Tool name identifier for NCBI"
    )

    # MinerU PDF Parsing
    MINERU_API_TOKEN: Optional[str] = Field(
        None,
        description="MinerU API token for PDF parsing"
    )
    MINERU_API_BASE: str = Field(
        "https://mineru.net/api/v4/extract/task",
        description="MinerU API endpoint"
    )
    MINERU_MODEL_VERSION: str = Field(
        "vlm",
        description="MinerU model version (pipeline or vlm)"
    )
    MINERU_MAX_POLL_TIME: int = Field(
        300,
        description="Maximum polling time for MinerU tasks (seconds)"
    )
    MINERU_POLL_INTERVAL: int = Field(
        5,
        description="Polling interval for MinerU tasks (seconds)"
    )

    # PDF Cache
    PDF_CACHE_DIR: str = Field(
        "/tmp/sl_agent/pdf_cache",
        description="Directory for caching downloaded PDFs"
    )

    # ==================== Retrieval Parameters ====================
    MIN_PAPERS: int = Field(
        3,
        description="Minimum number of papers to retrieve"
    )
    MAX_CHUNKS_PER_PAPER: int = Field(
        2,
        description="Maximum chunks to extract per paper"
    )
    SCORE_THRESHOLD: float = Field(
        0.3,
        description="Minimum relevance score threshold"
    )

    # ==================== Feature Flags ====================
    USE_ENTITY_BOOST: bool = Field(
        True,
        description="Enable NER-based entity boosting in retrieval"
    )
    USE_ROUND_ROBIN: bool = Field(
        True,
        description="Enable round-robin diversity strategy"
    )
    USE_CONTEXT_COMPRESSION: bool = Field(
        False,
        description="Enable LLM-based context compression"
    )
    COMPRESSION_STRATEGY: str = Field(
        "hybrid",
        description="Compression strategy: sentence_filter, hybrid, llm"
    )
    COMPRESSION_RATIO: float = Field(
        0.6,
        description="Target compression ratio (0.6 = keep 60%)"
    )
    USE_LLM_COMPRESSION: bool = Field(
        False,
        description="Use LLM for compression (slower but more accurate)"
    )

    # Graph RAG
    USE_GRAPH_RAG: bool = Field(
        True,
        description="Enable dynamic knowledge graph extraction"
    )
    TRIPLET_EXTRACTION_MAX_TOKENS: int = Field(
        512,
        description="Max tokens for GraphRAG triplet extraction (ev1 default: 512)"
    )
    TRIPLET_EXTRACTION_TIMEOUT_SECONDS: float = Field(
        65.0,
        description="Timeout for GraphRAG triplet extraction in seconds (ev1 default: 65.0)"
    )
    MAX_TRIPLET_CHUNKS: int = Field(
        3,
        description="Maximum chunks for triplet extraction"
    )
    MAX_GRAPH_HOPS: int = Field(
        2,
        description="Maximum hops for graph traversal"
    )
    MAX_GRAPH_CONTEXT_TOKENS: int = Field(
        800,
        description="Maximum tokens for graph context (~7% of total)"
    )
    USE_BIO_NORMALIZATION: bool = Field(
        True,
        description="Use MyGene/MyChem API for entity normalization"
    )
    MYGENE_TIMEOUT: float = Field(
        3.0,
        description="MyGene API timeout (seconds)"
    )
    STRING_DB_THRESHOLD: float = Field(
        0.7,
        description="STRING DB confidence threshold"
    )
    USE_STRING_VALIDATION: bool = Field(
        True,
        description="Validate relationships with STRING DB"
    )

    # Open Targets
    USE_OPENTARGETS_VALIDATION: bool = Field(
        True,
        description="Validate relationships with Open Targets"
    )
    OPENTARGETS_GRAPHQL_URL: str = Field(
        "https://api.platform.opentargets.org/api/v4/graphql",
        description="Open Targets GraphQL endpoint"
    )
    OPENTARGETS_SEARCH_URL: str = Field(
        "https://api.platform.opentargets.org/api/v4/search",
        description="Open Targets search endpoint"
    )
    OPENTARGETS_TIMEOUT: float = Field(
        5.0,
        description="Open Targets API timeout (seconds)"
    )
    OPENTARGETS_SCORE_THRESHOLD: float = Field(
        0.1,
        description="Minimum score threshold for Open Targets associations"
    )

    # Disease Keywords
    DISEASE_KEYWORDS: set = Field(
        default_factory=lambda: {
            'CANCER', 'TUMOR', 'CARCINOMA', 'SARCOMA', 'LEUKEMIA',
            'LYMPHOMA', 'MELANOMA', 'SYNDROME', 'DISEASE', 'DISORDER',
            'GLIOBLASTOMA', 'ADENOCARCINOMA', 'MYELOMA', 'NEUROBLASTOMA'
        },
        description="Disease keywords for heuristic filtering"
    )

    # ==================== DepMap & External KG ====================
    DEPMAP_DATA_DIR: str = Field(
        "/path/to/depmap/data",
        description="Directory containing DepMap CSV files"
    )
    DEPMAP_ESSENTIALITY_THRESHOLD: float = Field(
        -0.5,
        description="CERES score threshold for essentiality"
    )
    USE_DEPMAP_KB: bool = Field(
        True,
        description="Enable DepMap knowledge base injection"
    )

    # External KG (SQLite)
    EXTERNAL_KG_DB_PATH: str = Field(
        "/path/to/external/kg.db",
        description="Path to external knowledge graph SQLite database"
    )
    USE_EXTERNAL_KG: bool = Field(
        True,
        description="Enable external knowledge graph retrieval (P0 priority)"
    )
    EXTERNAL_KG_MAX_PATHS: int = Field(
        20,
        description="Maximum paths to return per query"
    )
    EXTERNAL_KG_HOPS: int = Field(
        1,
        description="Default traversal depth (1-hop or 2-hop)"
    )
    EXTERNAL_KG_PRIORITY_RELATIONS: List[str] = Field(
        default_factory=lambda: [
            'CelllinedependOnGene',
            'Drug_Drug_Interaction',
            'Gene_LowExpress_Anatomy'
        ],
        description="High-priority relationship types"
    )

    # ==================== Query Expansion ====================
    USE_GENE_SYNONYM_EXPANSION: bool = Field(
        True,
        description="Enable gene synonym expansion via MyGene.info"
    )
    MYGENE_MAX_ALIASES: int = Field(
        3,
        description="Maximum aliases per gene"
    )
    GENE_ALIAS_CACHE_SIZE: int = Field(
        1000,
        description="LRU cache size for gene aliases"
    )

    # ==================== HyDE ====================
    USE_HYDE: bool = Field(
        True,
        description="Enable Hypothetical Document Embeddings"
    )
    HYDE_TEMPERATURE: float = Field(
        0.9,
        description="Temperature for HyDE generation (ev1 default: 0.9)"
    )
    HYDE_MAX_TOKENS: int = Field(
        25600,
        description="Max tokens for HyDE generation (ev1 default: 25600)"
    )
    HYDE_TIMEOUT_SECONDS: float = Field(
        100.0,
        description="Timeout for HyDE generation in seconds (ev1 default: 100.0)"
    )
    HYDE_WEIGHT: float = Field(
        0.6,
        description="Weight for HyDE results in fusion"
    )
    RAW_QUERY_WEIGHT: float = Field(
        0.4,
        description="Weight for raw query results in fusion"
    )

    # ==================== Token Budget (Waterfall Strategy) ====================
    MAX_CONTEXT_TOKENS: int = Field(
        12000,
        description="Maximum total context tokens"
    )
    TOKEN_BUDGET_SYSTEM: int = Field(
        1000,
        description="Reserved tokens for system instructions"
    )
    TOKEN_BUDGET_EXTERNAL_KG_MAX: int = Field(
        1500,
        description="Maximum tokens for External KG (P0 priority)"
    )
    TOKEN_BUDGET_KB_MAX: int = Field(
        3000,
        description="Maximum tokens for Structured KB (P1 priority)"
    )
    TOKEN_BUDGET_GRAPH_MAX: int = Field(
        1500,
        description="Maximum tokens for Graph RAG (P2 priority)"
    )
    # P3 (Literature) gets remaining tokens dynamically

    # ==================== Evaluation Mode ====================
    EVAL_MODE: bool = Field(
        True,
        description="Enable evaluation mode (full reference content)"
    )
    EVAL_CONTENT_MAX_LENGTH: int = Field(
        10000,
        description="Maximum content length in eval mode (0 = unlimited)"
    )

    # ==================== Agentic RAG ====================
    USE_AGENTIC_RAG: bool = Field(
        True,
        description="Enable multi-hop agentic reasoning"
    )
    AGENTIC_SCORE_THRESHOLD: float = Field(
        0.4,
        description="Minimum score to trigger agentic mode"
    )
    AGENTIC_MAX_HOPS: int = Field(
        2,
        description="Maximum reasoning hops"
    )
    AGENTIC_MECHANISTIC_QUERIES: int = Field(
        3,
        description="Number of mechanistic queries per hop"
    )

    # ==================== Logging ====================
    LOG_FILE: Optional[str] = Field(
        None,
        description="Path to log file (None = stdout only)"
    )
    LOG_LEVEL: str = Field(
        "INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    # ==================== Server Configuration ====================
    HOST: str = Field(
        "0.0.0.0",
        description="Server host"
    )
    PORT: int = Field(
        16006,
        description="Server port"
    )
    WORKERS: int = Field(
        4,
        description="Number of worker processes (production)"
    )

    # ==================== Computed Properties ====================
    @property
    def vector_cache_path(self) -> str:
        """Auto-compute vector cache path from corpus file"""
        if self.VECTOR_CACHE:
            return self.VECTOR_CACHE
        return self.CORPUS_FILE.replace('.jsonl', '_vectors.pt')

    @property
    def depmap_crispr_file(self) -> str:
        """Path to DepMap CRISPR file"""
        import os
        return os.path.join(self.DEPMAP_DATA_DIR, "CRISPRGeneDependency.csv")

    @property
    def depmap_sl_benchmark_file(self) -> str:
        """Path to DepMap SL Benchmark file"""
        import os
        return os.path.join(self.DEPMAP_DATA_DIR, "SL_Benchmark_Final.csv")

    @property
    def entrez_delay(self) -> float:
        """Rate limit delay for NCBI Entrez API"""
        # With API key: 10 req/s, without: 3 req/s
        return 0.1 if self.NCBI_API_KEY else 0.34

    @property
    def EXTERNAL_KG_PATH(self) -> str:
        """Alias for EXTERNAL_KG_DB_PATH for compatibility"""
        return self.EXTERNAL_KG_DB_PATH

    @property
    def GRAPH_RAG_PATH(self) -> Optional[str]:
        """Path to graph RAG file"""
        import os
        graph_file = os.path.join(self.DEPMAP_DATA_DIR, "knowledge_graph.json")
        return graph_file if os.path.exists(graph_file) else None


# Global settings instance
settings = Settings()
