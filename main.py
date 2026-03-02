"""
Main Entry Point for SLAgent

FastAPI application with lifespan context for model loading.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes import router
from utils.logger import get_logger

# Models
from models import QwenEmbedder, QwenReranker, NERPipeline, ContextCompressor

# Retrieval
from retrieval import (
    DenseRetriever, SparseRetriever, HybridRetriever, EntityAnalyzer,
    ExternalKGRetriever, DepMapKBRetriever, GraphRAGRetriever
)

# Tools
from tools import (
    PubMedClient, PDFParser, WebCrawler,
    BioRxivClient, ClinicalTrialsClient, WebSearchClient
)

# Core
from core import Orchestrator, ContextAssembler, AgenticStateMachine

# LLM
from llm import LLMManager, IntentDetector, QueryGenerator, Solver, Verifier, HyDEGenerator, TokenCounter

logger = get_logger(__name__)


# Global instances (loaded at startup)
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context.

    Loads models and initializes components at startup,
    cleans up at shutdown.
    """
    logger.info("Starting SLAgent application")

    try:
        # Keep env-level proxy behavior aligned with ev1 for libraries that
        # rely on urllib/requests-style proxy discovery.
        if settings.PROXY_URL:
            os.environ["http_proxy"] = settings.PROXY_URL
            os.environ["https_proxy"] = settings.PROXY_URL
            os.environ["HTTP_PROXY"] = settings.PROXY_URL
            os.environ["HTTPS_PROXY"] = settings.PROXY_URL
            logger.info("Proxy environment variables injected", proxy=settings.PROXY_URL)

        # Initialize ML models
        logger.info("Loading ML models...")

        if settings.EMBEDDING_PATH:
            app_state['embedder'] = QwenEmbedder(
                model_path=settings.EMBEDDING_PATH,
                device=settings.DEVICE
            )
            logger.info("Embedder loaded")

        if settings.RERANKER_PATH:
            app_state['reranker'] = QwenReranker(
                model_path=settings.RERANKER_PATH,
                device=settings.DEVICE
            )
            logger.info("Reranker loaded")

        if settings.NER_MODEL_PATH:
            app_state['ner_pipeline'] = NERPipeline(
                model_path=settings.NER_MODEL_PATH,
                device=0 if 'cuda' in settings.DEVICE else -1
            )
            logger.info("NER pipeline loaded")

        # Initialize context compressor (query-aware mode uses embedder if available).
        app_state['compressor'] = ContextCompressor(
            embedder=app_state.get('embedder'),
            strategy=settings.COMPRESSION_STRATEGY,
            ratio=settings.COMPRESSION_RATIO
        )

        # Initialize tools
        logger.info("Initializing tools...")

        app_state['pubmed_client'] = PubMedClient(
            email=settings.NCBI_EMAIL,
            api_key=getattr(settings, 'NCBI_API_KEY', None)
        )

        app_state['pdf_parser'] = PDFParser(
            mineru_token=getattr(settings, 'MINERU_API_TOKEN', None)
        )

        app_state['biorxiv_client'] = BioRxivClient()

        app_state['clinical_trials_client'] = ClinicalTrialsClient()

        app_state['web_search'] = WebSearchClient(
            api_key=getattr(settings, 'SERPER_API_KEY', None)
        )

        # Initialize WebCrawler with routing to specialized clients
        app_state['web_crawler'] = WebCrawler()
        app_state['web_crawler'].set_clients(
            pubmed_client=app_state['pubmed_client'],
            biorxiv_client=app_state['biorxiv_client'],
            clinical_trials_client=app_state['clinical_trials_client'],
            pdf_parser=app_state['pdf_parser']
        )

        # Initialize retrievers
        logger.info("Initializing retrievers...")

        app_state['entity_analyzer'] = EntityAnalyzer(
            ner_pipeline=app_state.get('ner_pipeline')
        )

        app_state['external_kg'] = ExternalKGRetriever(
            db_path=getattr(settings, 'EXTERNAL_KG_PATH', None)
        )

        app_state['depmap_kb'] = DepMapKBRetriever(
            data_dir=getattr(settings, 'DEPMAP_DATA_DIR', None)
        )

        app_state['graph_rag'] = GraphRAGRetriever(
            graph_path=getattr(settings, 'GRAPH_RAG_PATH', None)
        )

        # Local retrieval stack: Dense -> Sparse -> Hybrid(RRF)
        app_state['dense_retriever'] = None
        app_state['sparse_retriever'] = None
        app_state['hybrid_retriever'] = None

        if app_state.get('embedder'):
            try:
                app_state['dense_retriever'] = DenseRetriever(
                    embedder=app_state['embedder'],
                    corpus_file=settings.CORPUS_FILE,
                    vector_cache=settings.vector_cache_path,
                    device=settings.DEVICE
                )
            except Exception as e:
                logger.warning("Dense retriever initialization failed", error=str(e))

        if app_state.get('dense_retriever'):
            try:
                app_state['sparse_retriever'] = SparseRetriever(
                    documents=app_state['dense_retriever'].documents,
                    metadatas=app_state['dense_retriever'].metadatas,
                    ner_pipeline=app_state.get('ner_pipeline')
                )
            except Exception as e:
                logger.warning("Sparse retriever initialization failed", error=str(e))

        # Initialize LLM components
        logger.info("Initializing LLM components...")

        app_state['llm_manager'] = LLMManager()
        app_state['intent_detector'] = IntentDetector(app_state['llm_manager'])
        app_state['query_generator'] = QueryGenerator(
            app_state['llm_manager'],
            entity_analyzer=app_state.get('entity_analyzer')
        )
        app_state['solver'] = Solver(app_state['llm_manager'])
        app_state['verifier'] = Verifier(app_state['llm_manager'])

        # Initialize HyDE Generator
        if settings.USE_HYDE:
            app_state['hyde_generator'] = HyDEGenerator(app_state['llm_manager'])
            logger.info("HyDE Generator initialized")

        # Build hybrid retriever after HyDE component is available
        if app_state.get('dense_retriever'):
            app_state['hybrid_retriever'] = HybridRetriever(
                dense_retriever=app_state['dense_retriever'],
                sparse_retriever=app_state.get('sparse_retriever'),
                hyde_generator=app_state.get('hyde_generator'),
                entity_analyzer=app_state.get('entity_analyzer')
            )
        else:
            logger.warning("Hybrid retriever disabled - dense retriever unavailable")

        # Set LLM client on Graph RAG for triplet extraction
        if 'graph_rag' in app_state and app_state['graph_rag'].is_available():
            try:
                llm_client = app_state['llm_manager'].get_client()
                app_state['graph_rag'].set_llm_client(
                    llm_client,
                    settings.SOLVER_MODEL_NAME
                )
                logger.info("Graph RAG LLM client set for triplet extraction")
            except Exception as e:
                logger.warning("Failed to set Graph RAG LLM client", error=str(e))

        # Set LLM client on Context Compressor for LLM-based compression
        if 'compressor' in app_state:
            try:
                llm_client = app_state['llm_manager'].get_client()
                app_state['compressor'].set_llm_client(
                    llm_client,
                    settings.SOLVER_MODEL_NAME
                )
                logger.info("Context Compressor LLM client set")
            except Exception as e:
                logger.warning("Failed to set Compressor LLM client", error=str(e))

        # Initialize core components
        app_state['token_counter'] = TokenCounter()
        app_state['context_assembler'] = ContextAssembler(
            token_counter=app_state['token_counter']
        )

        # Initialize Agentic State Machine with required components
        if settings.USE_AGENTIC_RAG and app_state.get('hybrid_retriever'):
            app_state['agentic_state_machine'] = AgenticStateMachine(
                query_generator=app_state['query_generator'],
                hybrid_retriever=app_state['hybrid_retriever'],
                web_search=app_state.get('web_search'),
                web_crawler=app_state.get('web_crawler'),
                reranker=app_state.get('reranker'),
                score_threshold=settings.AGENTIC_SCORE_THRESHOLD,
                max_hops=settings.AGENTIC_MAX_HOPS
            )
            logger.info("Agentic State Machine initialized")
        else:
            app_state['agentic_state_machine'] = None

        # Initialize orchestrator with all components
        app_state['orchestrator'] = Orchestrator(
            hybrid_retriever=app_state['hybrid_retriever'],
            entity_analyzer=app_state['entity_analyzer'],
            intent_detector=app_state['intent_detector'],
            query_generator=app_state['query_generator'],
            context_assembler=app_state['context_assembler'],
            solver=app_state['solver'],
            verifier=app_state['verifier'],
            agentic_state_machine=app_state['agentic_state_machine'],
            external_kg=app_state.get('external_kg'),
            depmap_kb=app_state.get('depmap_kb'),
            graph_rag=app_state.get('graph_rag'),
            web_search=app_state.get('web_search'),
            web_crawler=app_state.get('web_crawler'),
            reranker=app_state.get('reranker'),
            compressor=app_state.get('compressor')
        )
        app.state.orchestrator = app_state['orchestrator']

        logger.info("All components loaded successfully")

    except Exception as e:
        logger.error("Failed to initialize components", error=str(e))
        raise RuntimeError(f"Failed to initialize SLAgent: {e}") from e

    yield  # Application runs

    # Cleanup
    logger.info("Shutting down SLAgent application")

    # Close database connections
    if 'external_kg' in app_state:
        app_state['external_kg'].close()

    if 'llm_manager' in app_state:
        await app_state['llm_manager'].close()

    app_state.clear()


# Create FastAPI app
app = FastAPI(
    title="SLAgent",
    description="Synthetic Lethality RAG Backend for Precision Oncology",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SLAgent",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": bool(app_state)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # Development mode
        log_level=settings.LOG_LEVEL.lower()
    )
