import asyncio
import concurrent.futures
import datetime
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
import zipfile
from typing import Dict, Any, Tuple

import nltk
import numpy as np
import requests
import torch
import torch.nn.functional as F
import trafilatura
import urllib3

# Disable SSL warnings (to bypass certificate issues with some corporate firewalls)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# NetworkX for Graph RAG
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ networkx not installed. Graph RAG will be disabled.")
    print("   Install with: pip install networkx")

# httpx for async HTTP requests (Bio API normalization)
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️ httpx not installed. BioAPI normalization will be disabled.")
    print("   Install with: pip install httpx")
from io import BytesIO
from torch import Tensor

# BM25 for sparse retrieval
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank_bm25 not installed. Sparse retrieval will be disabled.")
    print("   Install with: pip install rank-bm25")

# PDF parsing
try:
    import pdfplumber

    # io module already imported at top level
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber not installed. PDF parsing will be disabled.")
    print("   Install with: pip install pdfplumber")

# Biopython for PubMed access
try:
    from Bio import Entrez

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("⚠️ Biopython not installed. PubMed abstract extraction will be disabled.")
    print("   Install with: pip install biopython")
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional

# FastAPI & Streaming
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI  # Use an asynchronous client

# ================= Configuration Area =================

# Local path
DATA_ROOT = os.getenv("SLAGENT_DATA_ROOT", "./data/slagent")
MODEL_ROOT = os.getenv("SLAGENT_MODEL_ROOT", os.path.join(DATA_ROOT, "models"))
EMBEDDING_PATH = os.getenv("EMBEDDING_PATH", os.path.join(MODEL_ROOT, "Qwen3-Embedding-0.6B"))
RERANKER_PATH = os.getenv("RERANKER_PATH", os.path.join(MODEL_ROOT, "Qwen3-Reranker-0.6B"))
CORPUS_FILE = os.getenv("CORPUS_FILE", os.path.join(DATA_ROOT, "miner", "rag_corpus_contextual.jsonl"))
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", os.path.join(MODEL_ROOT, "d4data", "biomedical-ner-all"))
VECTOR_CACHE = CORPUS_FILE.replace('.jsonl', '_vectors.pt')
DEVICE = "cuda:0"
_log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.getenv("LOG_DIR", os.path.join(DATA_ROOT, "vllm", "raglog"))
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, f"rag_system_{_log_timestamp}.log"))
# 2. API Configuration
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
API_KEY = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
# Note: If the backend is deployed on a server, ensure it can access this proxy; otherwise leave it empty.
PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:6152")
# Base URL compatible with the OpenAI format
SOLVER_API_BASE = os.getenv("SOLVER_API_BASE", "https://chat.intern-ai.org.cn/api/v1")
SOLVER_MODEL_NAME = os.getenv("SOLVER_MODEL_NAME", "intern-s1-mini")
VERIFIER_MODEL_NAME = os.getenv("VERIFIER_MODEL_NAME", "intern-s1-mini")

# NCBI Entrez configuration
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@example.com")
# Optional: Enter the NCBI API key to increase the rate limit to 10 requests per second
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_TOOL_NAME = "SLAgent_RAG"  # Tool name identifier

# MinerU API Configuration
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")
MINERU_API_BASE = "https://mineru.net/api/v4/extract/task"
MINERU_MODEL_VERSION = "vlm"  # Optional: "pipeline" or "vlm"
MINERU_MAX_POLL_TIME = 300  # Maximum polling time (seconds)
MINERU_POLL_INTERVAL = 5  # Polling interval (seconds)

# PDF cache configuration (used to bypass 403 restrictions)
PDF_CACHE_DIR = os.getenv("PDF_CACHE_DIR", os.path.join(DATA_ROOT, "vllm", "pdf_cache"))  # PDF temporary cache directory

# 3. Retrieval parameters
MIN_PAPERS = 3
MAX_CHUNKS_PER_PAPER = 2
SCORE_THRESHOLD = 0.3
USE_ENTITY_BOOST = True
USE_ROUND_ROBIN = True  # Whether to use the Round-Robin diversity strategy
MAX_CONTEXT_TOKENS = 12000

# 4. Context Compression Configuration
USE_CONTEXT_COMPRESSION = False  # Whether to enable the context compression feature
COMPRESSION_STRATEGY = "hybrid"  # Compression strategy: "sentence_filter" | "hybrid" | "llm"
COMPRESSION_RATIO = 0.6  # Target compression ratio (retain 60% of the content)
# Minimum similarity threshold between the sentence and the query
# Whether to use an LLM for compression (slower but more accurate). If True, enables a second-stage compression for strategy="llm" or hybrid.
USE_LLM_COMPRESSION = False

# 5. Graph RAG Configuration
USE_GRAPH_RAG = True  # Whether to enable Graph RAG
MAX_TRIPLET_CHUNKS = 3  # Maximum number of chunks for extracting triples
MAX_GRAPH_HOPS = 2  # Maximum number of hops for graph traversal
MAX_GRAPH_CONTEXT_TOKENS = 800  # Maximum token count for graph context (about 7% of 12000)
USE_BIO_NORMALIZATION = True  # Whether to use the MyGene/MyChem API for entity normalization
MYGENE_TIMEOUT = 3.0  # MyGene API request timeout (seconds)
STRING_DB_THRESHOLD = 0.7  # STRING DB validation confidence threshold
USE_STRING_VALIDATION = True  # Whether to use the STRING database to validate interactions (adds latency but is more accurate)

# 6. Open Targets Platform Configuration
USE_OPENTARGETS_VALIDATION = True  # Whether to enable Open Targets validation
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPENTARGETS_SEARCH_URL = "https://api.platform.opentargets.org/api/v4/search"
OPENTARGETS_TIMEOUT = 5.0  # GraphQL query timeout (seconds)
OPENTARGETS_SCORE_THRESHOLD = 0.1  # Associations below this score will be ignored.
# Disease keywords (used for heuristic filtering of Gene→Disease edges)
DISEASE_KEYWORDS = {'CANCER', 'TUMOR', 'CARCINOMA', 'SARCOMA', 'LEUKEMIA',
                    'LYMPHOMA', 'MELANOMA', 'SYNDROME', 'DISEASE', 'DISORDER',
                    'GLIOBLASTOMA', 'ADENOCARCINOMA', 'MYELOMA', 'NEUROBLASTOMA'}

# 7. DepMap local data configuration (for knowledge base injection)
DEPMAP_DATA_DIR = os.getenv("DEPMAP_DATA_DIR", os.path.join(DATA_ROOT, "SLbm", "data"))
DEPMAP_CRISPR_FILE = os.path.join(DEPMAP_DATA_DIR, "CRISPRGeneDependency.csv")
DEPMAP_SL_BENCHMARK_FILE = os.path.join(DEPMAP_DATA_DIR, "SL_Benchmark_Final.csv")
DEPMAP_ESSENTIALITY_THRESHOLD = -0.5  # CERES score threshold (below this value, it is considered essential)
USE_DEPMAP_KB = True  # Whether to enable DepMap knowledge base injection

# 7b. External knowledge graph configuration (SQLite database)
EXTERNAL_KG_DB_PATH = os.path.join(DEPMAP_DATA_DIR, "kg.db")  # SQLite database path
USE_EXTERNAL_KG = True  # Whether to enable an external knowledge graph (priority 0: highest confidence)
EXTERNAL_KG_MAX_PATHS = 20  # Maximum number of paths returned by a single query
EXTERNAL_KG_HOPS = 1  # Default traversal depth (1 hop or 2 hops)
EXTERNAL_KG_PRIORITY_RELATIONS = [
    'CelllinedependOnGene',  # DepMap necessity (highest priority)
    'Drug_Drug_Interaction',  # Drug interaction (safety-critical)
    'Gene_LowExpress_Anatomy'  # Low expression (for side effect prediction)
]
# Token budget: allocate 1500 tokens specifically for the external KG (P0, highest priority)
TOKEN_BUDGET_EXTERNAL_KG_MAX = 1500

# 8. Query extended configuration
USE_GENE_SYNONYM_EXPANSION = True  # Whether to enable gene synonym expansion
MYGENE_MAX_ALIASES = 3  # Maximum number of aliases that can be used per gene
GENE_ALIAS_CACHE_SIZE = 1000  # Gene alias cache size

# 9. HyDE Configuration
USE_HYDE = True  # Whether to enable HyDE (hypothetical document embeddings)
HYDE_WEIGHT = 0.6  # Weight of HyDE retrieval results (0.0–1.0)
RAW_QUERY_WEIGHT = 0.4  # Original query search result weight

# 10. Token Budget Allocation Strategy (Waterfall)
TOKEN_BUDGET_SYSTEM = 1000  # Reserve system instructions
TOKEN_BUDGET_KB_MAX = 3000  # Maximum token count for the structured knowledge base (P1 priority)
TOKEN_BUDGET_GRAPH_MAX = 1500  # Maximum token limit for Graph RAG (P2 priority)
# Allocate all remaining space to literature search (P3 priority).

# 11. Configure evaluation mode
EVAL_MODE = True  # Whether to enable evaluation mode (return the full reference content without truncation)
EVAL_CONTENT_MAX_LENGTH = 10000  # Maximum content length in evaluation mode (0 means no limit)

# 12. Agent RAG (multi-hop reasoning) configuration
USE_AGENTIC_RAG = True  # Whether to enable Agentic RAG multi-hop reasoning
AGENTIC_SCORE_THRESHOLD = 0.4  # Minimum score threshold for triggering multi-hop reasoning
AGENTIC_MAX_HOPS = 2  # Maximum inference hops
AGENTIC_MECHANISTIC_QUERIES = 3  # Number of templated queries generated each time

# 4. Prompts
# Intent classification prompt
INTENT_PROMPT = """
Classify the following User Query into one of two categories:
1. "GENERAL": Casual conversation, greetings, weather, jokes, or general questions unrelated to science/biology (e.g., "Hi", "How are you", "What is the capital of France").
2. "SCIENTIFIC": Questions related to biology, medicine, genetics, cancer, synthetic lethality, research papers, or specific technical concepts.

User Query: "{query}"

Reply ONLY with "GENERAL" or "SCIENTIFIC". Do not add punctuation.
"""

# General small-talk and guidance prompt words
GENERAL_CHAT_PROMPT = """
You are an expert AI Assistant specializing in **Synthetic Lethality (SL) and Precision Oncology**.
The user has initiated a general conversation or greeting: "{query}"

Your Task:
1. Answer the user's input politely and briefly (e.g., return the greeting, answer the general question).
2. **Crucially**, steer the conversation back to your expertise. Invite the user to ask about Synthetic Lethality, cancer targets, drug mechanisms, or recent papers.

Example:
User: "Hello"
AI: "Hello! I'm ready to help. I specialize in analyzing Synthetic Lethality targets for cancer therapy. Do you have a specific gene or pathway you'd like to investigate today?"
"""

QUERY_GEN_PROMPT = """
### Role
You are an **Expert Scientific Search Engine Optimizer** specializing in **Synthetic Lethality (SL)** and Precision Oncology. You act as a "Research Navigator" to bridge the gap between raw genetic data and actionable therapeutic targets.

### Task
Generate **4 distinct, highly optimized Google search queries** based on the User Question: "{question}".
Your goal is to mimic the search behavior of a senior principal investigator (PI) verifying a new SL target.

### Strategy & Chain-of-Thought (CoT)
1.  **Seminal Validation (The "Source of Truth"):**
    * *Think*: What is the original paper that defined this SL pair? Is it a robust finding in top-tier journals?
    * *Operators*: Use `"synthetic lethality" OR "collateral lethality"` AND `("Nature" OR "Cell" OR "Science" OR "Cancer Discovery")` to find the landmark study.
2.  **Mechanistic Context & Dependency:**
    * *Think*: SL is often context-dependent (e.g., only in p53-null background). How does it work? (Metabolism, DNA Repair, Immuno-oncology).
    * *Operators*: Use `"mechanism of action"` OR `"metabolic vulnerability"` OR `"context dependent"`.
3.  **Clinical Translation & Inhibitors:**
    * *Think*: Are there small molecule inhibitors or ADCs targeting this? What phase is the trial?
    * *Operators*: Use `site:clinicaltrials.gov` OR `("Phase I" OR "Phase II")` AND `"inhibitor"`.
4.  **Frontier & Resistance (2024-2025):**
    * *Think*: What are the latest resistance mechanisms or novel combinations published *this year*?
    * *Operators*: Use `2024..2025` AND (`"resistance mechanism"` OR `"novel target"` OR `site:biorxiv.org`).

### Few-Shot Example
**User Question**: "Target for MTAP-deleted glioblastoma"
**Output**:
"MTAP loss" "PRMT5" OR "MAT2A" "synthetic lethality" nature OR cell OR science
"MTAP deficiency" mechanism "accumulation of MTA" OR "metabolic vulnerability"
site:clinicaltrials.gov "MTAP" AND ("PRMT5 inhibitor" OR "MAT2A inhibitor")
"MTAP" synthetic lethal "resistance" OR "combination therapy" 2024..2025 filetype:pdf

### User Question
"{question}"

### Output Format
Return ONLY the 4 queries, one per line. No introductory text, no markdown bullets.
"""

# A step-back prompt for Agentic RAG (multi-hop reasoning)
STEP_BACK_PROMPT = """
The user asked about synthetic lethality for: '{query}'

Direct retrieval yielded LOW-CONFIDENCE results (max score < 0.4).
This suggests sparse data on direct SL evidence.

**Your Task**: Step back and identify the BIOLOGICAL CONTEXT to enable indirect reasoning.

**Chain-of-Thought Analysis:**
1. **Pathway Membership**: What cellular pathway does the gene belong to?
   - Examples: Homologous Recombination (HR), Base Excision Repair (BER), Glycolysis, Mitochondrial Function
   - Keywords: "DNA repair pathway", "metabolic pathway", "signaling cascade"

2. **Paralog Redundancy**: Are there known paralogs or functionally redundant partners?
   - Examples: BRCA1/BRCA2, PARP1/PARP2, ATM/ATR
   - Keywords: "paralog", "redundant function", "backup pathway"

3. **Stress Signature**: What happens when this gene is deleted?
   - Examples: "replication stress", "oxidative stress", "metabolic crisis"
   - Keywords: "loss of function", "cellular stress", "vulnerability"

4. **Known Synthetic Lethal Partners**: Are there EXISTING validated SL pairs for genes in the SAME pathway?
   - Examples: "BRCA1 + PARP1", "VHL + HIF"
   - Keywords: "synthetic lethal with", "collateral lethality", "context-specific dependency"

**Output Format:**
Generate {num_queries} mechanistic search queries (NOT the original question) focused on:
- Pathway function and compensatory mechanisms
- Paralog relationships and functional redundancy
- Cellular stress phenotypes induced by gene loss
- Known SL relationships within the same biological context

**Example:**
Original Query: "synthetic lethal partner for MTAP deletion"
Output:
1. "MTAP metabolic pathway polyamine synthesis salvage"
2. "MTAP loss MAT2A PRMT5 dependency mechanism"
3. "methionine salvage pathway redundancy MTAP ADI-PEG20"

Return ONLY the {num_queries} queries, one per line. NO explanations.
"""

SOLVER_PROMPT = """
...
**CRITICAL INSTRUCTION ON DATA SOURCES:**
1. **[External Knowledge Graph]** sections contain PURE FACTUAL DATA from databases like DepMap and DrugBank. Treat these as GROUND TRUTH.
   - If the KG says "CellLine X depends on Gene Y", it is a verified fact, not a hallucination.
2. **[Retrieved Literature]** sections contain unstructured text from papers. Use them to explain the *mechanisms* behind the facts found in the KG.
...
You are an expert oncologist specializing in Synthetic Lethality (SL).

**1. Definition & Scope**
Begin by defining Synthetic Lethality (SL) as a genetic interaction where the perturbation of two genes/pathways leads to cell death, while the perturbation of either alone is viable.
* **Scope:** Include **Direct Genetic Interactions** (e.g., hard-wired physical dependencies) and **Indirect Mechanisms** (e.g., compensatory pathway redundancy).
* **Indirect Effects:** You must acknowledge and analyze indirect drivers such as **epigenetic modulation, pathway crosstalk, and metabolic rewiring**, provided they are supported by mechanistic hypotheses or functional genomics data.

**2. Tiered Evidence Framework & Scoring**
Prioritize hypotheses using the following hierarchy. **Final Ranking = Evidence Tier × Mechanistic Plausibility Score (1–3).**

* **Tier 1 (Direct Empirical):** Validated interactions from CRISPR screens (e.g., DepMap), co-dependency datasets, or high-quality functional studies.
* **Tier 2 (Preclinical Synergy):** Drug combination studies showing SL in biologically relevant tumor lineages.
* **Tier 3 (Mechanistic Inference):**
    * **3a (Pathway-Aligned):** Hypotheses derived from the gene’s canonical function (e.g., DNA repair defects creating dependency on backup repair pathways, like BRCA1/PARP).
    * **3b (Stress-Response Driven):** Hypotheses where **context-specific stress** (e.g., acidosis, hypoxia, replication stress) forces reliance on a compensatory pathway.
        * *Criteria:* Prioritize if the mutation impairs stress adaptation, making the cell addicted to a survival kinase (e.g., mTOR, ATR) or specific ion transporter.
    * **3c (Metabolic Rewiring):** Hypotheses based on metabolic bottlenecks.
        * *Example:* Dysregulated PTMs (e.g., SUMOylation defects via SENP1 loss) stabilizing metabolic drivers (e.g., HIF1α), creating dependencies on glycolysis, angiogenesis, or specific nutrient transporters.

**3. Target Evaluation Strategy**
When evaluating a specific gene mutation for SL targets:

* **Integrate Functional Genomics:** Query (mentally or via tools) databases like DepMap/GDSC. Look for statistically significant correlations between the mutation and drug sensitivity (e.g., "Does Mutation X correlate with sensitivity to ALK or MEK inhibitors in broad screens?").
* **Lineage & Microenvironment:** If the exact lineage is unknown, infer plausible contexts based on mutation prevalence. explicitly model **microenvironmental stressors**:
    * *pH/Acidosis:* For genes regulating pH (e.g., CA2), hypothesize that loss-of-function leads to intracellular/extracellular acidosis. This may activate **acid-sensing GPCRs or stress-responsive RTKs** (e.g., AXL, EGFR, or TRK families) or sensitize cells to ROS-inducing agents.
* **Map Gene Function to Vulnerability:**
    * *Canonical:* Loss of function → Dependency on paralog/backup.
    * *Non-Canonical:* Gain of function/structure → Dependency on chaperone proteins or downstream effectors.
* **Evolutionary Pressure:** Consider if the mutation promotes polyploidy or aneuploidy, creating dependencies on mitotic checkpoints (e.g., Aurora Kinases, PLK1).

**4. Experimental Validation Plan**
Propose **discriminatory experiments** tailored to the mechanism:

* **Kinase/Signaling:** Western blots for phospho-proteins in mutant vs. WT cells under relevant stress (e.g., p-AKT, p-ERK under hypoxia).
* **DNA Repair/Replication:** Rad51 foci formation, Comet assays, or R-loop quantification.
* **Metabolic:** Metabolomics profiling (e.g., lactate production), Seahorse assays, or rescue experiments with specific metabolites.
* **Context-Specific Screens:** **Crucial:** Propose performing CRISPR screens or drug sensitivity assays under **physiologically relevant conditions** (e.g., acidic pH 6.5, low glucose, or hypoxia) to reveal "conditional SL" that is missed in standard culture.

**5. Clinical Translation & Comparison**
Structure the final recommendation by comparing options based on:
1.  **Interaction Strength** (Is it a "hard" kill or "soft" growth inhibition?)
2.  **Mechanistic Plausibility** (Does it violate biological dogma? Is the signaling link established?)
3.  **Translational Reality** (Is there an FDA-approved drug or specific inhibitor available? Can we repurpose a drug?)

**Response Structure:**
1.  **Mechanism:** Describe the molecular chain of events (e.g., "Gene A mutation leads to accumulation of Metabolite B, rendering cells hypersensitive to inhibition of Pathway C").
2.  **Evidence Assessment:** Assign Tier (1-3) and Plausibility Score.
3.  **Validation Plan:** Specific assays (in vitro & in vivo).
4.  **Clinical Context:** Drug repurposing potential (e.g., "Use Drug X, approved for Indication Y, in this novel context").
5.  **Alternative Scenarios:** Use probabilistic language (e.g., "While Option A is the primary hypothesis due to Tier 1 data, Option B warrants testing if the tumor exhibits hypoxic features").

**Examples of "Non-Canonical" Mechanisms to Consider:**
* **Metabolic Stabilization:** SENP1 loss leading to HIF1α stabilization (via failure to de-SUMOylate), driving VEGF/Glut1 expression and creating dependency on angiogenesis inhibitors or glycolysis blockade.
* **Non-Genomic Signaling:** Estrogen activating PI3K/AKT/MAPK pathways via **membrane-associated receptors (e.g., GPER)**, creating survival dependencies even in the absence of classical nuclear ER transcriptional activity.
* **pH-Dependent Signaling:** Acidic microenvironments (due to CA2 or transport defects) constitutively activating stress-survival kinases (e.g., NF-κB or specific RTKs), making them viable targets only under in vivo conditions.

**Output the final answer as a scientifically rigorous, ranked hypothesis.**
"""

VERIFIER_PROMPT = """
You are a verification system for a Clinical Decision Task. Your task is BINARY: output ONLY "VERDICT: PASS" or "VERDICT: FAIL".

### **Checklist**
1.  **Format Compliance**: Does the Candidate Answer end with or consist strictly of "Answer: [A/B/C/D]"?
2.  **Logic Consistency**:
    * Does the selected option represent a **Synthetic Lethal (SL)** strategy (Targeting Gene B given Mutation in Gene A)?
    * Did the solver avoid choosing the "Direct Targeting" distractor (Targeting Gene A given Mutation in Gene A) unless it was the only logical choice?
3.  **Accuracy** (if context provided): Is the selected pair (Mutation -> Drug Target) supported by the provided context or general biological consensus?

### **Output Rules**
* If the answer follows the format and logic: Output "VERDICT: PASS"
* If the format is wrong or the logic explicitly fails (e.g., selects a random gene with no biological basis): Output "VERDICT: FAIL"

EXAMPLES:

Input: "CSK Mutation... Choices: A) Inhibitor of CSK... C) Inhibitor of GRIK1..."
Candidate: "Answer: C"
Output: VERDICT: PASS
(Reasoning: Correctly identified SL partner GRIK1 instead of direct target CSK)

Input: "..."
Candidate: "The answer is likely C because..." (No strict format)
Output: VERDICT: FAIL
"""

# Global resource container
global_resources = {}


# ================= Logging System =================


def log_to_file(message, level="INFO", console=False):
    """Write the log to a file, optionally also print to the console.
Args:
    message: Log message
    level: Log level
    console: Whether to also output to the console"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_message = f"[{timestamp}] [{level}] {message}\n"
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_message)

        # Output key logs to the console as well
        if console or any(
                kw in message for kw in ['✅ Crawl success', '❌ Crawl failed', '⏱️ Timeout', '⛔ 403', '⚠️', 'Fallback']):
            print(f"    {message}")
    except Exception as e:
        print(f"Failed to write log: {e}")


# Helper class definitions (Embedder, Reranker)


def get_gpu_memory_usage(device):
    if device.startswith('cuda'):
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 3
        return allocated, reserved
    return 0, 0


class QwenEmbedder:
    def __init__(self, model_path, device):
        print(f"Loading Embedding: {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side='left')
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, texts, is_query=False, task_instruction=""):
        log_to_file(f"[Embedder] Start encoding: is_query={is_query}, text count={len(texts)}")

        if is_query:
            input_texts = [
                f'Instruct: {task_instruction}\nQuery:{q}' for q in texts]
            log_to_file(f"[Embedder] Query mode, Instruction:{task_instruction}")
            log_to_file(f"[Embedder] Query text:{texts[0][:100]}...")
        else:
            input_texts = texts
            log_to_file(
                f"[Embedder] Document mode, first text:{texts[0][:100] if texts else 'None'}...")

        batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True,
                                    truncation=True, return_tensors="pt").to(self.device)
        log_to_file(
            f"[Embedder] Shape after tokenization:{batch_dict['input_ids'].shape}")

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        log_to_file(f"[Embedder] Output embedding shape:{embeddings.shape}")
        return embeddings


class QwenReranker:
    def __init__(self, model_path, device):
        print(f"Loading Reranker: {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)

    def compute_score(self, query, document, instruction=None, domain="general"):
        """Compute the query–document relevance score.

Args:
    query: User query
    document: Candidate document
    instruction: Custom instruction (if None, automatically selected based on domain)
    domain: Domain type ("synthetic_lethality" | "general")

Returns:
    Relevance score [0.0, 1.0]"""
        # Select instruction based on domain
        if instruction is None:
            if domain == "synthetic_lethality":
                instruction = """Given a query about Synthetic Lethality or Cancer Therapy, retrieve documents describing:
(1) Gene-gene synthetic lethal interactions or functional dependencies
(2) Small molecule inhibitors targeting SL partners or compensatory pathways  
(3) Clinical trials exploiting SL vulnerabilities (e.g., PARP inhibitors in BRCA-deficient cancers)
(4) Mechanistic explanations of genetic dependencies, pathway rewiring, or drug resistance
Prioritize documents with gene names, drug mechanisms, essentiality scores, or experimental evidence.
Ignore general descriptions of genes without interaction details."""
            else:
                instruction = "Given a web search query, retrieve relevant passages that answer the query"

        # Only record the first 100 characters to avoid overly long logs
        log_to_file(
            f"[Reranker] Domain: {domain}, Query: {query[:100]}..., Doc: {document[:100]}...")

        raw_text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        input_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
        original_length = len(input_ids)

        input_ids = input_ids[:8192 -
                               len(self.prefix_tokens) - len(self.suffix_tokens)]
        final_input_ids = self.prefix_tokens + input_ids + self.suffix_tokens
        input_tensor = torch.tensor([final_input_ids], device=self.device)

        log_to_file(
            f"[Reranker] Input tokens: {original_length} -> {len(final_input_ids)} (truncated)")

        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits[0, -1]
            yes_logit = logits[self.token_yes_id]
            no_logit = logits[self.token_no_id]
            score = torch.exp(yes_logit) / \
                    (torch.exp(yes_logit) + torch.exp(no_logit))

        score_value = score.item()
        log_to_file(
            f"[Reranker] Score: {score_value:.4f} (yes_logit={yes_logit:.2f}, no_logit={no_logit:.2f})")
        return score_value


class ContextCompressor:
    """Context compressor  
Integrated: NLTK sentence segmentation, batch embedding, context window expansion, L2 normalization"""
    # Download NLTK data (compatible with both old and new versions)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            nltk.download('punkt', quiet=True)

    def __init__(self, embedder, tokenizer, device, async_client=None,
                 strategy="sentence_filter", ratio=0.5, context_window=1):
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.device = device
        self.async_client = async_client
        self.strategy = strategy
        self.ratio = ratio
        self.context_window = context_window  # Added: context window size (retain a certain number of preceding and following sentences)

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ContextCompressor")
        self.logger.info(
            f"✅ Init - Strategy: {strategy}, Ratio: {ratio}, Window: {context_window}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """More accurate sentence segmentation using NLTK"""
        if not text:
            return []
        # Use NLTK's `sent_tokenize`
        sentences = nltk.sent_tokenize(text)
        # Filter out overly short noise segments (e.g., "Fig 1.")
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _filter_sentences_with_window(self, sentences: List[str], scores: torch.Tensor) -> str:
        """Core logic: filter based on scores, and apply contextual window expansion  
Args:  
    sentences: list of sentences  
    scores: similarity score for each sentence (Tensor)"""
        if len(sentences) <= 3:
            return " ".join(sentences)

        num_sentences = len(sentences)
        # Calculate the number of core sentences to retain
        topk = max(1, int(num_sentences * self.ratio))

        # Get the indices of the top k highest-scoring sentences
        # torch.topk returns (values, indices)
        _, top_indices = torch.topk(scores, k=min(topk, num_sentences))
        top_indices = top_indices.cpu().tolist()

        # Use a Set to store the indices of sentences to keep in the final result (deduplicated automatically).
        indices_to_keep = set()

        for idx in top_indices:
            # Window expansion logic: keep the interval [idx - window, idx + window]
            start = max(0, idx - self.context_window)
            end = min(num_sentences - 1, idx + self.context_window)

            # Add the indices within the range to the set.
            for i in range(start, end + 1):
                indices_to_keep.add(i)

        # Sort the indices to ensure reconstruction follows the original order.
        sorted_indices = sorted(list(indices_to_keep))

        # Reorganize the text
        compressed_text = " ".join([sentences[i] for i in sorted_indices])
        return compressed_text

    async def _llm_compress(self, query: str, content: str) -> str:
        """LLM Compression - Prompt-Optimized Version"""
        if not self.async_client:
            return content

        # Optimized prompt: emphasize “Verbatim” (word-for-word) and “No Paraphrasing” (no rewriting).
        prompt = f"""You are a precise data compressor. 
Task: Extract VERBATIM segments from the Document that answer the Query.
Constraint: Do NOT rewrite or paraphrase. Keep the original text structure. 
If the document contains no relevant info, return an empty string.

Query: {query}
Document: {content}

Compressed Output:"""

        try:
            response = await self.async_client.chat.completions.create(
                model=SOLVER_MODEL_NAME,  # Use the configured model.
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower the temperature to reduce hallucinations.
                max_tokens=1024
            )
            compressed = response.choices[0].message.content
            self.logger.info(
                f"LLM compression: {len(content)} -> {len(compressed)} chars")
            return compressed
        except Exception as e:
            self.logger.error(f"LLM compression failed: {e}")
            return content

    async def compress_chunks(self, query: str, chunks: List[dict]) -> List[dict]:
        """Batch-compress the retrieved chunks (core implementation of Batch Processing)"""
        if not chunks or self.strategy == "none":
            return chunks

        self.logger.info(
            f"Starting compression for {len(chunks)} chunks (Batch Mode)")

        # Preprocessing: collect all sentences from all chunks
        all_sentences = []
        # Record mapping relationships: [{ 'start': 0, 'end': 5, 'original_chunk': chunk_obj }, ...]
        chunk_map = []

        for chunk in chunks:
            # Only split sentences for the `sentence_filter` and `hybrid` strategies.
            if self.strategy in ["sentence_filter", "hybrid"]:
                sents = self._split_into_sentences(chunk["content"])
                start_idx = len(all_sentences)
                all_sentences.extend(sents)
                end_idx = len(all_sentences)

                chunk_map.append({
                    "start": start_idx,
                    "end": end_idx,
                    "sentences": sents,
                    "original": chunk
                })
            else:
                # For now, handle the LLM or extractive strategy according to the existing logic, or handle it separately.
                chunk_map.append({"original": chunk, "skip_batch": True})

        # 2. Batch embedding (batch computation)
        # If there are no sentences requiring similarity calculation (e.g., all are LLM strategy), then skip.
        if all_sentences:
            # Encode the Query (embedder.encode has already returned a normalized Tensor)
            query_emb = self.embedder.encode([query])
            if query_emb.device != self.device:
                query_emb = query_emb.to(self.device)

            # Encode all sentences at once
            # Note: If the number of sentences is very large (>1000), you may need to use a mini-batch loop here, depending on GPU memory.
            sent_embs = self.embedder.encode(all_sentences)
            if sent_embs.device != self.device:
                sent_embs = sent_embs.to(self.device)

            # Compute the similarity between all sentences and the query; output shape is [1, Total_Sents].
            all_scores = torch.mm(query_emb, sent_embs.T).squeeze(0)

        # 3. Distribute back to each block and perform compression
        compressed_chunks = []

        for idx, item in enumerate(chunk_map):
            chunk = item["original"]

            # Strategy branch
            if item.get("skip_batch"):
                # Strategies that do not rely on embeddings (e.g., pure LLM)
                if self.strategy == "llm" and USE_LLM_COMPRESSION:
                    compressed_content = await self._llm_compress(query, chunk["content"])
                else:
                    compressed_content = chunk["content"]

            else:
                # Embedded dependency strategy
                start = item["start"]
                end = item["end"]
                sents = item["sentences"]

                if start == end:  # Empty sentence list
                    compressed_content = chunk["content"]
                else:
                    # Get the score corresponding to this Chunk via slicing
                    chunk_scores = all_scores[start:end]

                    if self.strategy == "sentence_filter":
                        compressed_content = self._filter_sentences_with_window(
                            sents, chunk_scores)
                    elif self.strategy == "hybrid":
                        # Hybrid mode: perform embedding-based filtering first
                        compressed_content = self._filter_sentences_with_window(
                            sents, chunk_scores)
                        # Optional: secondary LLM compression (further improve quality)
                        if USE_LLM_COMPRESSION and len(compressed_content) > 500:
                            compressed_content = await self._llm_compress(query, compressed_content)
                    else:
                        compressed_content = chunk["content"]

            # Build result
            new_chunk = chunk.copy()
            new_chunk["content"] = compressed_content
            new_chunk["original_length"] = len(chunk["content"])
            new_chunk["compressed_length"] = len(compressed_content)
            compressed_chunks.append(new_chunk)

        return compressed_chunks


# ================= External Knowledge Graph RAG (SQLite-based) =================

class KnowledgeGraphRAG:
    """High-performance knowledge graph retrieval engine based on SQLite  
Data source: kg.db (converted from raw_kg.tsv with 12M+ triples)

Advantages:  
1. **Zero load time** - plug and play, no need to wait for data loading  
2. **Ultra-low memory** - < 10MB memory usage, supports hundreds of GB of data  
3. **Millisecond-level queries** - high-speed retrieval backed by B-Tree indexes  
4. **Bidirectional queries** - supports forward (x->y) and reverse (y<-x) relation queries

Confidence: P0 (highest priority) - structured data, more reliable than text search"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.available = False
        self.stats = {'total_rows': 0, 'unique_relations': 0}

        if not USE_EXTERNAL_KG:
            log_to_file("[ExternalKG] DISABLED by config")
            return

        if not os.path.exists(self.db_path):
            log_to_file(f"[ExternalKG] Database not found: {self.db_path}", "WARNING")
            log_to_file("[ExternalKG] Please run build_kg_db.py first to convert TSV to SQLite", "WARNING")
            return

        try:
            # Test the connection and retrieve statistics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM relationships')
            self.stats['total_rows'] = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT relation) FROM relationships')
            self.stats['unique_relations'] = cursor.fetchone()[0]

            conn.close()

            self.available = True
            log_to_file(f"[ExternalKG] ✅ Connected to database: {self.db_path}")
            log_to_file(f"[ExternalKG]    - Total relationships: {self.stats['total_rows']:,}")
            log_to_file(f"[ExternalKG]    - Unique relation types: {self.stats['unique_relations']}")

        except Exception as e:
            log_to_file(f"[ExternalKG] ❌ Failed to connect: {e}", "ERROR")
            import traceback
            log_to_file(traceback.format_exc(), "ERROR")

    def _get_connection(self):
        """Obtain a database connection (thread-safe: one independent connection per request)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Allow access by column name
        return conn

    def query_subgraph(self, entities: List[str], hops: int = None) -> str:
        """Query an entity’s 1-hop or 2-hop relations as needed

Args:
    entities: List of gene or drug names (e.g., ['MTAP', 'PRMT5', 'Olaparib'])
    hops: Traversal depth (None = use the configured default)

Returns:
    Formatted graph fact text for injecting into the LLM context"""
        if not self.available or not entities:
            return ""

        if hops is None:
            hops = EXTERNAL_KG_HOPS

        # Standardize entity names
        entities = [e.strip().upper() for e in entities if e]
        if not entities:
            return ""

        log_to_file(f"[ExternalKG] Querying {len(entities)} entities: {entities[:5]}")

        conn = self._get_connection()
        cursor = conn.cursor()

        found_paths = set()
        priority_paths = set()

        # High-priority relationship (used for sorting)
        PRIORITY_RELS_UPPER = {rel.upper() for rel in EXTERNAL_KG_PRIORITY_RELATIONS}

        try:
            for entity in entities:
                # === Forward query: x -> relation -> y ===
                cursor.execute("""
                               SELECT x_name, relation, display_relation, y_name, y_type, rel_source
                               FROM relationships
                               WHERE x_name = ? LIMIT 50
                               """, (entity,))

                rows = cursor.fetchall()

                for row in rows:
                    path = f"- {row['x_name']} → {row['display_relation']} → {row['y_name']} (Type: {row['y_type']}, Source: {row['rel_source']})"

                    # Priority determination
                    if row['relation'].upper() in PRIORITY_RELS_UPPER:
                        priority_paths.add(path)
                    else:
                        found_paths.add(path)

                    # === Two-hop query: x -> y -> z ===
                    if hops > 1:
                        target_node = row['y_name']
                        cursor.execute("""
                                       SELECT relation, display_relation, y_name, y_type
                                       FROM relationships
                                       WHERE x_name = ? LIMIT 10
                                       """, (target_node,))

                        sub_rows = cursor.fetchall()
                        for sub in sub_rows:
                            # Pruning: avoid returning to the starting point
                            if sub['y_name'] == entity:
                                continue

                            path_2hop = f"- {row['x_name']} → {row['display_relation']} → {target_node} → {sub['display_relation']} → {sub['y_name']}"

                            if sub['relation'].upper() in PRIORITY_RELS_UPPER:
                                priority_paths.add(path_2hop)
                            else:
                                found_paths.add(path_2hop)

                # Reverse lookup: y ← relation ← x (extremely important!)
                # For example: query which cell lines depend on this gene
                cursor.execute("""
                               SELECT x_name, x_type, display_relation, y_name, rel_source, relation
                               FROM relationships
                               WHERE y_name = ? LIMIT 50
                               """, (entity,))

                back_rows = cursor.fetchall()
                for row in back_rows:
                    path = f"- {row['x_name']} ({row['x_type']}) → {row['display_relation']} → {row['y_name']} (Source: {row['rel_source']})"

                    # Reverse dependencies are usually important (e.g., CellLine → depends on → Gene).
                    if row['relation'].upper() in PRIORITY_RELS_UPPER:
                        priority_paths.add(path)
                    else:
                        found_paths.add(path)

        except Exception as e:
            log_to_file(f"[ExternalKG] Query error: {e}", "ERROR")
        finally:
            conn.close()

        # Result merge: higher-priority paths first
        all_paths = list(priority_paths) + list(found_paths)

        if not all_paths:
            log_to_file(f"[ExternalKG] No paths found for entities: {entities}")
            return ""

        # Limit the number of returned items
        limited_paths = all_paths[:EXTERNAL_KG_MAX_PATHS]

        # Format it into natural language.
        result = f"=== [External Knowledge Graph - {len(limited_paths)} Facts] ===\n"
        result += f"Data Source: DepMap, DrugBank, Bgee (Anatomical Expression)\n"
        result += f"Confidence Level: P0 (Highest - Structured Data)\n"
        result += f"Query: {', '.join(entities)}\n\n"
        result += "\n".join(limited_paths)

        log_to_file(
            f"[ExternalKG] Found {len(limited_paths)} paths "
            f"({len(priority_paths)} high-priority, {len(found_paths)} standard)"
        )

        return result

    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """Get detailed information for a single node

Returns:
    {'exists': bool, 'out_degree': int, 'in_degree': int, 'type': str}"""
        if not self.available:
            return {'exists': False}

        node_upper = node_name.strip().upper()

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Count out-degree (x -> y)
            cursor.execute('SELECT COUNT(*) FROM relationships WHERE x_name = ?', (node_upper,))
            out_degree = cursor.fetchone()[0]

            # Compute in-degree (y <- x)
            cursor.execute('SELECT COUNT(*) FROM relationships WHERE y_name = ?', (node_upper,))
            in_degree = cursor.fetchone()[0]

            if out_degree == 0 and in_degree == 0:
                return {'exists': False}

            # Get the node type
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
            log_to_file(f"[ExternalKG] get_node_info error: {e}", "ERROR")
            return {'exists': False}
        finally:
            conn.close()


# ================= Knowledge Base Augmenter =================

class KnowledgeBaseAugmenter:
    """Local knowledge base enhancer — used to inject SL knowledge from structured databases such as DepMap  
Functions:  
1. Load local DepMap CRISPR essentiality data  
2. Load validated SL benchmark pairs  
3. Generate synthetic documents for retrieval augmentation  
4. Provide fast KB validation functionality"""

    def __init__(self):
        self.crispr_data = None
        self.sl_benchmark = None
        self.gene_essentiality_cache = {}  # {gene: {cell_line: score}}
        self.sl_pairs_cache = set()  # {(gene_a, gene_b)}

        if USE_DEPMAP_KB:
            self._load_depmap_data()
            log_to_file("[KB] KnowledgeBaseAugmenter initialized")
        else:
            log_to_file("[KB] KnowledgeBaseAugmenter: DISABLED by config")

    def _load_depmap_data(self):
        """Load local DepMap data"""
        try:
            # Load SL benchmarks (validated SL pairs)
            if os.path.exists(DEPMAP_SL_BENCHMARK_FILE):
                import pandas as pd
                self.sl_benchmark = pd.read_csv(DEPMAP_SL_BENCHMARK_FILE)
                # Build a set for fast lookups
                for _, row in self.sl_benchmark.iterrows():
                    gene_a = str(row.get('gene_a', '')).upper()
                    gene_b = str(row.get('gene_b', '')).upper()
                    if gene_a and gene_b:
                        # Bidirectional storage (A→B and B→A)
                        self.sl_pairs_cache.add((gene_a, gene_b))
                        self.sl_pairs_cache.add((gene_b, gene_a))
                log_to_file(f"[KB] Loaded {len(self.sl_benchmark)} SL benchmark pairs")

            # Load CRISPR gene dependency (essentiality scores)
            # Note: This file may be large (>100MB); load as needed.
            if os.path.exists(DEPMAP_CRISPR_FILE):
                import pandas as pd
                # Only load the first 1000 rows as a sample (in production, load all data or use a database)
                self.crispr_data = pd.read_csv(DEPMAP_CRISPR_FILE, nrows=1000)
                log_to_file(f"[KB] Loaded CRISPR data: {self.crispr_data.shape[0]} cell lines (sample)")

        except Exception as e:
            log_to_file(f"[KB] Failed to load DepMap data: {e}", "ERROR")

    def get_sl_pairs(self, gene: str) -> List[Dict[str, Any]]:
        """Query known SL partners of a given gene"""
        if self.sl_benchmark is None or not USE_DEPMAP_KB:
            return []

        gene_upper = gene.upper()
        results = []

        # Find all synthetic lethal gene pairs (SL pairs) that include this gene.
        matched = self.sl_benchmark[
            (self.sl_benchmark['gene_a'].str.upper() == gene_upper) |
            (self.sl_benchmark['gene_b'].str.upper() == gene_upper)
            ]

        for _, row in matched.iterrows():
            gene_a = row.get('gene_a', '')
            gene_b = row.get('gene_b', '')
            partner = gene_b if gene_a.upper() == gene_upper else gene_a

            results.append({
                'partner': partner,
                'evidence_source': row.get('evidence_source', 'Unknown'),
                'cell_line': row.get('cell_line', 'N/A'),
                'pubmed_id': row.get('pubmed_id', ''),
                'cancer_type': row.get('cancer_type', ''),
                'benchmark_level': row.get('benchmark_level', '')
            })

        return results

    def augment_with_depmap(self, genes: List[str]) -> List[Dict[str, Any]]:
        """Generate synthetic documents for the retrieved genes. These documents are inserted into the retrieval candidates to improve recall in sparse-data scenarios."""
        if not USE_DEPMAP_KB or not genes or self.sl_benchmark is None:
            return []

        synthetic_docs = []

        for gene in genes:
            gene_upper = gene.upper()

            # Query known SL pairings
            sl_pairs = self.get_sl_pairs(gene_upper)
            if sl_pairs:
                # Generate an SL pairing summary document
                partners = [p['partner'] for p in sl_pairs[:5]]  # Only take the first 5.
                evidence_sources = set(p['evidence_source'] for p in sl_pairs)

                content = f"[DepMap Knowledge Base - Synthetic Lethality]\n\n"
                content += f"Gene: {gene_upper}\n"
                content += f"Known Synthetic Lethal Partners: {', '.join(partners)}\n\n"

                for pair in sl_pairs[:3]:  # Provide a detailed description of the first three.
                    content += f"- {gene_upper} is synthetic lethal with {pair['partner']}\n"
                    content += f"  Evidence Source: {pair['evidence_source']}\n"
                    if pair['cell_line'] and pair['cell_line'] != 'N/A':
                        content += f"  Cell Line: {pair['cell_line']}\n"
                    if pair['pubmed_id']:
                        content += f"  PubMed ID: {pair['pubmed_id']}\n"
                    content += "\n"

                content += f"Total validated SL interactions: {len(sl_pairs)}\n"
                content += f"Data sources: {', '.join(evidence_sources)}"

                synthetic_docs.append({
                    'content': content,
                    'metadata': {
                        'paper_title': f"DepMap SL Database: {gene_upper} Interactions",
                        'link': f"depmap://sl_benchmark/{gene_upper}",
                        'source': 'DepMap_KB',
                        'date': '2024',
                        'is_synthetic': True,  # Marked as a synthesized document
                        'key_genes': [gene_upper] + partners[:3],  # Includes the main gene and the first three partners.
                        'key_methods': ['synthetic_lethality', 'depmap'],
                        'is_web': False
                    }
                })

        if synthetic_docs:
            log_to_file(f"[KB] Generated {len(synthetic_docs)} synthetic documents from DepMap")

        return synthetic_docs

    def validate_sl_pair(self, gene_a: str, gene_b: str) -> Dict[str, Any]:
        """Validate whether the two genes are a known SL pair.

Returns:
    Dict with keys: is_validated, evidence, confidence"""
        if not USE_DEPMAP_KB:
            return {'is_validated': False, 'evidence': None, 'confidence': 0.0}

        gene_a_upper = gene_a.upper()
        gene_b_upper = gene_b.upper()

        # Check the cache
        if (gene_a_upper, gene_b_upper) in self.sl_pairs_cache or \
                (gene_b_upper, gene_a_upper) in self.sl_pairs_cache:
            # Obtain detailed evidence
            pairs = self.get_sl_pairs(gene_a_upper)
            for p in pairs:
                if p['partner'].upper() == gene_b_upper:
                    return {
                        'is_validated': True,
                        'evidence': f"{p['evidence_source']} ({p['benchmark_level']})",
                        'confidence': 0.9 if 'Gold' in str(p['benchmark_level']) else 0.7,
                        'pubmed_id': p['pubmed_id']
                    }

        return {'is_validated': False, 'evidence': None, 'confidence': 0.0}


# ================= Graph RAG Manager =================

class SimpleGraphManager:
    """Lightweight memory graph manager (Graph-on-Retrieved-Context)
Functions: extract triples from retrieved text -> build a NetworkX graph -> retrieve multi-hop relations
Integration: MyGene.info gene normalization + STRING DB relation validation (optional)"""

    # Triple extraction prompt (minimal format, must use pipe separators)
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

    def __init__(self, async_client, model_name: str, tokenizer=None):
        """Initialize the Graph Manager.

Args:
    async_client: AsyncOpenAI client
    model_name: Model name used for triple extraction
    tokenizer: Tokenizer used for token counting"""
        self.async_client = async_client
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.graph = None  # nx.DiGraph: rebuilt for each query

        # Entity normalization cache (persistent across requests, reduces API calls)
        self.normalization_cache: Dict[str, str] = {}

        # Ensembl ID cache (gene symbol -> Ensembl ID, for Open Targets queries)
        self.ensembl_id_cache: Dict[str, str] = {}

        # EFO ID cache (disease name -> EFO ID)
        self.efo_cache: Dict[str, str] = {}

        # Open Targets validation results cache ((gene, disease) -> result string)
        self.ot_cache: Dict[Tuple[str, str], str] = {}

        # Hard-coded mapping of common gene aliases (fast fallback)
        self.gene_alias_map = {
            "p53": "TP53", "P53": "TP53",
            "her2": "ERBB2", "HER2": "ERBB2", "neu": "ERBB2",
            "ras": "KRAS", "k-ras": "KRAS",
            "arf": "CDKN2A", "p14arf": "CDKN2A", "p16": "CDKN2A",
            "myc": "MYC", "c-myc": "MYC",
            "egfr": "EGFR", "her1": "EGFR",
        }

        log_to_file("[GraphRAG] SimpleGraphManager initialized")

    async def extract_triplets(self, text_chunks: List[dict]) -> List[Tuple[str, str, str]]:
        """Extract triples from text concurrently using an LLM

Args:
    text_chunks: list of dictionaries containing a 'content' key

Returns:
    List of (subject, relation, object) tuples"""
        # Only process the first N chunks to avoid excessive latency and API rate limiting
        target_chunks = text_chunks[:MAX_TRIPLET_CHUNKS]

        if not target_chunks:
            log_to_file("[GraphRAG] No chunks to extract triplets from")
            return []

        log_to_file(f"[GraphRAG] Extracting triplets from {len(target_chunks)} chunks")

        async def _extract_single(text: str) -> List[Tuple[str, str, str]]:
            """Triple extraction for a single chunk"""
            try:
                # Filter out overly short text
                if len(text) < 50:
                    return []

                # Truncate the input to prevent it from being too long.
                truncated_text = text[:1500]

                response = await asyncio.wait_for(
                    self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{
                            "role": "user",
                            "content": self.TRIPLET_EXTRACTION_PROMPT.format(text=truncated_text)
                        }],
                        temperature=0.0,  # Must be 0 to ensure deterministic output.
                        max_tokens=512  # Limit the output length
                    ),
                    timeout=65.0  # Single request timeout
                )

                raw_output = response.choices[0].message.content
                triplets = []

                # Parse triples separated by pipe characters
                for line in raw_output.strip().split('\n'):
                    line = line.strip()
                    if '|' in line and line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3 and all(parts):
                            triplets.append((parts[0], parts[1], parts[2]))

                log_to_file(f"[GraphRAG] Extracted {len(triplets)} triplets from chunk")
                return triplets

            except asyncio.TimeoutError:
                log_to_file("[GraphRAG] Triplet extraction timeout", "WARNING")
                return []
            except Exception as e:
                log_to_file(f"[GraphRAG] Triplet extraction error: {e}", "ERROR")
                return []

        # Concurrently extract all chunks
        tasks = [_extract_single(chunk.get('content', '')) for chunk in target_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten the results and filter out anomalies
        all_triplets = []
        for res in results:
            if isinstance(res, list):
                all_triplets.extend(res)

        log_to_file(f"[GraphRAG] Total triplets extracted: {len(all_triplets)}")
        return all_triplets

    async def normalize_nodes_with_mygene(self, entities: List[str]) -> Dict[str, str]:
        """Use the MyGene.info API to batch-normalize gene names

Args:
    entities: List of entities to be normalized

Returns:
    Dict mapping original names to normalized names"""
        if not entities or not USE_BIO_NORMALIZATION or not HTTPX_AVAILABLE:
            # Use a hardcoded mapping directly, and fall back to the uppercase form.
            result = {}
            for e in entities:
                e_lower = e.lower()
                if e_lower in self.gene_alias_map:
                    result[e] = self.gene_alias_map[e_lower]
                else:
                    result[e] = e.upper()
            return result

        # Filter cached entities
        unknown_entities = [e for e in set(entities) if e not in self.normalization_cache]

        if not unknown_entities:
            return {e: self.normalization_cache.get(e, e.upper()) for e in entities}

        log_to_file(f"[GraphRAG] MyGene batch normalizing {len(unknown_entities)} entities")

        url = "https://mygene.info/v3/query"
        payload = {
            "q": unknown_entities,
            "scopes": "symbol,alias,entrezgene,uniprot",
            "fields": "symbol,ensembl.gene",  # Simultaneously retrieve the Ensembl ID for Open Targets
            "species": "human",  # Humans only; extremely important
            "dotfield": False
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=MYGENE_TIMEOUT)
                results = resp.json()

            # Parsing result
            for item in results:
                if isinstance(item, dict):
                    original = item.get('query')
                    official = item.get('symbol')

                    # [NEW] Extract Ensembl IDs (the structure may be a list or a dict)
                    ens_data = item.get('ensembl')
                    ensembl_id = None
                    if isinstance(ens_data, list) and ens_data:
                        ensembl_id = ens_data[0].get('gene')
                    elif isinstance(ens_data, dict):
                        ensembl_id = ens_data.get('gene')

                    if original:
                        if official:
                            self.normalization_cache[original] = official
                            # [Added] Cache Ensembl IDs for Open Targets use
                            if ensembl_id:
                                self.ensembl_id_cache[official] = ensembl_id
                                log_to_file(f"[GraphRAG] Cached Ensembl ID: {official} -> {ensembl_id}")
                        else:
                            # Not found (possibly a drug or mechanism); please use a hard-coded value or ALL CAPS.
                            o_lower = original.lower()
                            if o_lower in self.gene_alias_map:
                                self.normalization_cache[original] = self.gene_alias_map[o_lower]
                            else:
                                self.normalization_cache[original] = original.upper()

            log_to_file(
                f"[GraphRAG] MyGene normalized {len(unknown_entities)} entities, cache size: {len(self.normalization_cache)}")

        except Exception as e:
            log_to_file(f"[GraphRAG] MyGene API failed: {e}. Falling back to uppercase.", "WARNING")
            # Failure fallback: use hardcoded mapping or convert to uppercase
            for e in unknown_entities:
                e_lower = e.lower()
                if e_lower in self.gene_alias_map:
                    self.normalization_cache[e] = self.gene_alias_map[e_lower]
                else:
                    self.normalization_cache[e] = e.upper()

        return {e: self.normalization_cache.get(e, e.upper()) for e in entities}

    async def verify_interaction_with_string(self, gene_a: str, gene_b: str) -> str:
        """Use the STRING DB to verify the interaction between two genes.

Args:
    gene_a: The first gene
    gene_b: The second gene

Returns:
    Verification status string (e.g., "[Verified: 0.95]" or "[Unverified]")"""
        if not USE_STRING_VALIDATION or not HTTPX_AVAILABLE:
            return ""

        identifiers = f"{gene_a}%0d{gene_b}"
        url = f"https://string-db.org/api/json/network?identifiers={identifiers}&species=9606"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=2.0)
                data = resp.json()

            if data and len(data) > 0:
                score = data[0].get('score', 0)
                if score >= STRING_DB_THRESHOLD:
                    return f"[Verified: {score:.2f}]"

        except Exception as e:
            log_to_file(f"[GraphRAG] STRING DB verification failed: {e}", "WARNING")

        return "[Unverified]"

    async def _get_efo_id(self, disease_name: str) -> str:
        """Retrieve a disease EFO ID via the Open Targets Search API.

Args:
    disease_name: Disease name (e.g., "Glioblastoma")

Returns:
    EFO ID (e.g., "EFO_0000519") or None"""
        # Check the cache
        if disease_name in self.efo_cache:
            return self.efo_cache[disease_name]

        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    OPENTARGETS_SEARCH_URL,
                    params={"q": disease_name, "filter": "disease", "size": 1},
                    timeout=2.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('hits'):
                        efo_id = data['hits'][0]['id']
                        self.efo_cache[disease_name] = efo_id
                        log_to_file(f"[GraphRAG] Resolved EFO ID: {disease_name} -> {efo_id}")
                        return efo_id
        except Exception as e:
            log_to_file(f"[GraphRAG] EFO ID lookup failed for {disease_name}: {e}", "WARNING")

        return None

    async def verify_opentargets(self, gene_symbol: str, disease_name: str) -> str:
        """Validate gene–disease associations using the Open Targets GraphQL API

Args:
    gene_symbol: Gene symbol (e.g., "MTAP")
    disease_name: Disease name (e.g., "Glioblastoma")

Returns:
    Validation result string or None"""
        if not USE_OPENTARGETS_VALIDATION or not HTTPX_AVAILABLE:
            return None

        # 0. Check cache
        cache_key = (gene_symbol, disease_name)
        if cache_key in self.ot_cache:
            return self.ot_cache[cache_key]

        # Retrieve the Ensembl ID (must be obtained in the MyGene step).
        ensembl_id = self.ensembl_id_cache.get(gene_symbol)
        if not ensembl_id:
            log_to_file(f"[GraphRAG] No Ensembl ID for {gene_symbol}, skipping OT verification")
            return None

        # 2. Obtain the EFO ID
        efo_id = await self._get_efo_id(disease_name)
        if not efo_id:
            log_to_file(f"[GraphRAG] No EFO ID for {disease_name}, skipping OT verification")
            return None

        # 3. GraphQL Query
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

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    OPENTARGETS_GRAPHQL_URL,
                    json={
                        "query": query,
                        "variables": {"ensemblId": ensembl_id, "efoIds": [efo_id]}
                    },
                    timeout=OPENTARGETS_TIMEOUT
                )
                data = resp.json()

                rows = data.get('data', {}).get('target', {}).get('associatedDiseases', {}).get('rows', [])

                result_str = None
                if rows:
                    score = rows[0]['score']
                    if score >= OPENTARGETS_SCORE_THRESHOLD:
                        # Parse the specific evidence source (choose the highest-scoring evidence type).
                        datatype_scores = rows[0].get('datatypeScores', [])
                        top_evidence = "N/A"
                        if datatype_scores:
                            top = max(datatype_scores, key=lambda x: x.get('score', 0))
                            top_evidence = top.get('id', 'unknown').replace('_', ' ').title()

                        credibility = "High" if score > 0.5 else "Moderate"
                        result_str = f"OT Validation: {gene_symbol}-{disease_name} (Score: {score:.2f}, {credibility}, Top Evidence: {top_evidence})"
                        log_to_file(f"[GraphRAG] {result_str}")

                # Update cache
                self.ot_cache[cache_key] = result_str
                return result_str

        except Exception as e:
            log_to_file(f"[GraphRAG] OpenTargets query failed: {e}", "WARNING")
            return None

    def build_graph(self, triplets: List[Tuple[str, str, str]], norm_map: Dict[str, str]) -> int:
        """Build a NetworkX directed graph

Args:
    triplets: list of triplets
    norm_map: entity normalization mapping

Returns:
    number of edges in the graph"""
        if not NETWORKX_AVAILABLE:
            log_to_file("[GraphRAG] NetworkX not available, skipping graph building", "WARNING")
            return 0

        self.graph = nx.DiGraph()

        for s, r, o in triplets:
            # Perform normalization.
            s_norm = norm_map.get(s, s.upper())
            o_norm = norm_map.get(o, o.upper())

            # Filter out self-loops
            if s_norm != o_norm:
                self.graph.add_edge(s_norm, o_norm, relation=r)

        edge_count = self.graph.number_of_edges()
        node_count = self.graph.number_of_nodes()
        log_to_file(f"[GraphRAG] Graph built: {node_count} nodes, {edge_count} edges")

        return edge_count

    def get_graph_context(self, query_entities: Dict[str, set], depth: int = 2) -> str:
        """Retrieve multi-hop paths from the graph that are related to the query entities.

Args:
    query_entities: A dictionary containing 'genes' and 'keywords' sets
    depth: Traversal depth

Returns:
    A formatted knowledge string from the graph"""
        if not NETWORKX_AVAILABLE or self.graph is None or self.graph.number_of_edges() == 0:
            return ""

        # Collect queried entities
        entities = set()
        for gene in query_entities.get("genes", set()):
            entities.add(gene.upper())
        for kw in list(query_entities.get("keywords", set()))[:3]:  # Limit the number of keywords
            entities.add(kw.upper())

        if not entities:
            return ""

        # Build an all-uppercase mapping of graph nodes
        graph_nodes_upper = {n.upper(): n for n in self.graph.nodes()}

        # Find matching seed nodes
        target_nodes = []
        for q_ent in entities:
            for node_upper, original_node in graph_nodes_upper.items():
                if q_ent in node_upper or node_upper in q_ent:
                    target_nodes.append(original_node)

        target_nodes = list(set(target_nodes))

        if not target_nodes:
            log_to_file(f"[GraphRAG] No matching nodes found for entities: {entities}")
            return ""

        log_to_file(f"[GraphRAG] Found {len(target_nodes)} seed nodes: {target_nodes[:5]}")

        # BFS traversal to collect relationships
        relevant_triplets = set()

        for node in target_nodes:
            # Outgoing edges (Node -> X)
            if node in self.graph:
                for neighbor in self.graph[node]:
                    rel = self.graph[node][neighbor].get('relation', 'related to')
                    relevant_triplets.add(f"{node} --[{rel}]--> {neighbor}")

                    # 2-hop
                    if depth > 1 and neighbor in self.graph:
                        for next_neighbor in self.graph[neighbor]:
                            rel2 = self.graph[neighbor][next_neighbor].get('relation', 'related to')
                            relevant_triplets.add(f"{neighbor} --[{rel2}]--> {next_neighbor}")

            # Incoming edge (X -> Node)
            try:
                for pred in self.graph.predecessors(node):
                    rel = self.graph[pred][node].get('relation', 'related to')
                    relevant_triplets.add(f"{pred} --[{rel}]--> {node}")

                    # two-hop in-degree
                    if depth > 1:
                        for pred2 in self.graph.predecessors(pred):
                            rel2 = self.graph[pred2][pred].get('relation', 'related to')
                            relevant_triplets.add(f"{pred2} --[{rel2}]--> {pred}")
            except:
                pass

        if not relevant_triplets:
            return ""

        # Formatted output
        triplet_list = sorted(list(relevant_triplets))
        graph_context = "=== KNOWLEDGE GRAPH INSIGHTS ===\n"
        graph_context += "(Extracted entity relationships from retrieved documents)\n\n"
        graph_context += "\n".join(triplet_list)

        log_to_file(f"[GraphRAG] Generated graph context with {len(triplet_list)} relationships")

        return graph_context

    async def build_and_query_graph(self, text_chunks: List[dict], query_entities: Dict[str, set]) -> str:
        """Main entry: extract triplets -> normalize -> build graph -> query

Args:
    text_chunks: retrieved text chunks
    query_entities: entities identified in the query

Returns:
    formatted graph context string"""
        if not NETWORKX_AVAILABLE:
            return ""

        try:
            # Extract triples
            triplets = await self.extract_triplets(text_chunks)

            if not triplets:
                log_to_file("[GraphRAG] No triplets extracted, skipping graph")
                return ""

            # 2. Collect all nodes
            all_nodes = set()
            for s, r, o in triplets:
                all_nodes.add(s)
                all_nodes.add(o)

            # 3. Batch normalization
            norm_map = await self.normalize_nodes_with_mygene(list(all_nodes))

            # 4. Build the graph
            edge_count = self.build_graph(triplets, norm_map)

            if edge_count == 0:
                return ""

            # Open Targets validation trigger (on-demand validation of gene–disease edges)
            ot_validations = []
            if USE_OPENTARGETS_VALIDATION and edge_count > 0:
                # Identify core genes (retrieved from query_entities)
                core_genes = {g.upper() for g in query_entities.get('genes', [])}

                verify_tasks = []
                seen_pairs = set()

                for u, v in self.graph.edges():
                    u_upper, v_upper = u.upper(), v.upper()

                    # Logic: the source node is a core gene, and the target node is a suspected disease.
                    is_core_gene = u_upper in core_genes
                    is_disease = any(dw in v_upper for dw in DISEASE_KEYWORDS)

                    # Avoid duplication
                    pair_key = f"{u_upper}-{v_upper}"

                    if is_core_gene and is_disease and pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        # Add a validation task
                        verify_tasks.append(self.verify_opentargets(u, v))

                # Execute concurrently (limit to the first 3 to ensure speed)
                if verify_tasks:
                    log_to_file(f"[GraphRAG] Verifying {len(verify_tasks)} potential disease associations")
                    results = await asyncio.gather(*verify_tasks[:3], return_exceptions=True)
                    for res in results:
                        if isinstance(res, str) and res:
                            ot_validations.append(res)

            # 6. Retrieve graph context
            graph_context = self.get_graph_context(query_entities, depth=MAX_GRAPH_HOPS)

            # 7. Additional Open Targets validation results
            if ot_validations and graph_context:
                graph_context += "\n\n[Gold Standard Validation (Open Targets)]\n" + "\n".join(ot_validations)

            return graph_context

        except Exception as e:
            log_to_file(f"[GraphRAG] build_and_query_graph failed: {e}", "ERROR")
            return ""

    def _count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in the text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Rough estimate: 4 characters are approximately equal to 1 token
        return len(text) // 4


# ================= RAG Logic Encapsulation =================


class RAGLogic:
    def __init__(self):
        self.embedder = QwenEmbedder(EMBEDDING_PATH, DEVICE)
        self.reranker = QwenReranker(RERANKER_PATH, DEVICE)
        self.docs_text, self.docs_meta, self.docs_vecs = self._load_data()
        self.tokenizer = self.embedder.tokenizer
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Build a BM25 index (sparse retrieval)
        if BM25_AVAILABLE:
            print("🔍 Building BM25 index for local corpus...")
            log_to_file("[Init] Starting BM25 index construction")
            # Perform simple tokenization on the document (split by whitespace and punctuation).
            tokenized_corpus = [self._tokenize_for_bm25(
                doc) for doc in self.docs_text]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"✅ BM25 index built: {len(tokenized_corpus)} documents")
            log_to_file(
                f"[Init] BM25 index: {len(tokenized_corpus)} documents indexed")
        else:
            self.bm25 = None
            print("⚠️ BM25 disabled: using Dense retrieval only")
            log_to_file("[Init] BM25: DISABLED (library not available)")

        print(f"Loading NER Model...")
        try:
            ner_device = int(DEVICE.split(":")[-1]) if "cuda" in DEVICE else -1
        except:
            ner_device = -1
        self.ner_pipeline = pipeline(
            "ner", model=NER_MODEL_PATH, tokenizer=NER_MODEL_PATH, aggregation_strategy="simple", device=ner_device)

        # Asynchronous LLM client for query generation
        self.async_client = AsyncOpenAI(
            api_key=API_KEY, base_url=SOLVER_API_BASE)

        # Initialize the context compressor
        self.compressor = ContextCompressor(
            embedder=self.embedder,
            tokenizer=self.tokenizer,
            device=DEVICE,
            async_client=self.async_client,
            strategy=COMPRESSION_STRATEGY,
            ratio=COMPRESSION_RATIO,
            context_window=1  # Context window: keep 1 sentence before and 1 sentence after each key sentence
        )

        # Initialize the Graph RAG manager
        if USE_GRAPH_RAG and NETWORKX_AVAILABLE:
            self.graph_manager = SimpleGraphManager(
                async_client=self.async_client,
                model_name=SOLVER_MODEL_NAME,
                tokenizer=self.tokenizer
            )
            print("✅ Graph RAG Manager: INITIALIZED")
            log_to_file("[Init] Graph RAG Manager: INITIALIZED")
        else:
            self.graph_manager = None
            if not USE_GRAPH_RAG:
                print("⚠️ Graph RAG: DISABLED (USE_GRAPH_RAG=False)")
                log_to_file("[Init] Graph RAG: DISABLED by config")
            else:
                print("⚠️ Graph RAG: DISABLED (NetworkX not available)")
                log_to_file("[Init] Graph RAG: DISABLED (missing networkx)")

        # Initialize the knowledge base augmenter
        self.kb_augmenter = KnowledgeBaseAugmenter()
        if USE_DEPMAP_KB:
            print("✅ Knowledge Base Augmenter: INITIALIZED (DepMap)")
            log_to_file("[Init] KB Augmenter: INITIALIZED")
        else:
            print("⚠️ Knowledge Base Augmenter: DISABLED")
            log_to_file("[Init] KB Augmenter: DISABLED by config")

        # Initialize the external knowledge graph (SQLite) — P0 priority (highest confidence)
        if USE_EXTERNAL_KG:
            self.external_kg = KnowledgeGraphRAG(EXTERNAL_KG_DB_PATH)
            if self.external_kg.available:
                print(f"✅ External Knowledge Graph: CONNECTED (SQLite)")
                print(f"   ├─ Database: {EXTERNAL_KG_DB_PATH}")
                print(f"   ├─ Relationships: {self.external_kg.stats['total_rows']:,}")
                print(f"   └─ Relation types: {self.external_kg.stats['unique_relations']}")
                log_to_file(
                    f"[Init] External KG: Connected with {self.external_kg.stats['total_rows']:,} relationships")
            else:
                print("⚠️ External Knowledge Graph: DATABASE NOT FOUND")
                print("   Run build_kg_db.py to convert raw_kg.tsv to SQLite first")
                log_to_file("[Init] External KG: Database not available", "WARNING")
                self.external_kg = None
        else:
            self.external_kg = None
            print("⚠️ External Knowledge Graph: DISABLED by config")
            log_to_file("[Init] External KG: DISABLED by config")

        # Gene alias cache (for query expansion)
        self.gene_alias_cache: Dict[str, List[str]] = {}
        if USE_GENE_SYNONYM_EXPANSION and HTTPX_AVAILABLE:
            print("✅ Gene Synonym Expansion: ENABLED")
            log_to_file("[Init] Gene Synonym Expansion: ENABLED")
        else:
            if not USE_GENE_SYNONYM_EXPANSION:
                print("⚠️ Gene Synonym Expansion: DISABLED by config")
                log_to_file("[Init] Gene Synonym Expansion: DISABLED by config")
            else:
                print("⚠️ Gene Synonym Expansion: DISABLED (httpx not available)")
                log_to_file("[Init] Gene Synonym Expansion: DISABLED (missing httpx)")

        # Configure NCBI Entrez (using Biopython if available)
        if BIOPYTHON_AVAILABLE:
            Entrez.email = NCBI_EMAIL
            Entrez.tool = NCBI_TOOL_NAME

            # Proxy configuration support: ensure Biopython's urllib can use the system proxy.
            # Biopython uses Python’s standard library urllib, which automatically reads environment variables.
            # But you need to ensure that both https_proxy and http_proxy are set.
            import os
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
            proxy_set = any(os.environ.get(var) for var in proxy_vars)
            if proxy_set:
                proxy_value = next((os.environ.get(var) for var in proxy_vars if os.environ.get(var)), None)
                print(f"🌐 Proxy detected: {proxy_value}")
                log_to_file(f"[Init] Proxy environment detected: {proxy_value}")
                # Ensure that https_proxy is also set (the PubMed API uses HTTPS).
                if not os.environ.get('https_proxy') and not os.environ.get('HTTPS_PROXY'):
                    if os.environ.get('http_proxy'):
                        os.environ['https_proxy'] = os.environ['http_proxy']
                    elif os.environ.get('HTTP_PROXY'):
                        os.environ['https_proxy'] = os.environ['HTTP_PROXY']
                    elif os.environ.get('all_proxy'):
                        os.environ['https_proxy'] = os.environ['all_proxy']
                        os.environ['http_proxy'] = os.environ['all_proxy']

            if NCBI_API_KEY:
                Entrez.api_key = NCBI_API_KEY
                print(f"✅ Biopython Entrez configured: {NCBI_EMAIL}")
                print(f"🔑 NCBI API Key: CONFIGURED (Rate: 10 req/sec)")
                log_to_file("[Init] NCBI API Key: CONFIGURED")
                log_to_file("[Init] Rate Limiting: 0.11s/request (10 req/sec)")
            else:
                print(f"✅ Biopython Entrez configured: {NCBI_EMAIL}")
                print(f"⚠️ NCBI API Key: NOT CONFIGURED (Rate: 3 req/sec)")
                log_to_file("[Init] NCBI API Key: NOT CONFIGURED")
                log_to_file("[Init] Rate Limiting: 0.34s/request (3 req/sec)")
            # Rate limiting: 3 req/sec without key, 10 req/sec with key
            self.entrez_delay = 0.11 if NCBI_API_KEY else 0.34
        else:
            self.entrez_delay = None
            print("⚠️ Biopython not available - PubMed API disabled")
            log_to_file("[Init] Biopython: NOT AVAILABLE")

        # Display PDF parsing status
        if MINERU_API_TOKEN:
            print(f"✅ MinerU API: CONFIGURED (Model: {MINERU_MODEL_VERSION})")
            log_to_file(
                f"[Init] MinerU API: CONFIGURED (Model: {MINERU_MODEL_VERSION})")
        else:
            print("⚠️ MinerU API: NOT CONFIGURED (will use pdfplumber fallback)")
            log_to_file("[Init] MinerU API: NOT CONFIGURED")

        # Initialize the PDF cache directory
        if not os.path.exists(PDF_CACHE_DIR):
            os.makedirs(PDF_CACHE_DIR, exist_ok=True)
            log_to_file(f"[Init] Created PDF cache directory: {PDF_CACHE_DIR}")
        else:
            log_to_file(f"[Init] PDF cache directory exists: {PDF_CACHE_DIR}")

        if PDFPLUMBER_AVAILABLE:
            print("✅ PDF parsing fallback: pdfplumber enabled")
            log_to_file("[Init] PDF Parsing Fallback: ENABLED (pdfplumber)")
        else:
            print("⚠️ PDF parsing fallback: pdfplumber disabled")
            log_to_file("[Init] PDF Parsing Fallback: DISABLED")

        # Check Serper API health status
        print("🔍 Testing Serper API...")
        if self._test_serper_api():
            print("✅ Serper API: OPERATIONAL")
        else:
            print("⚠️ Serper API: UNAVAILABLE (web search may fail)")

    def _parse_pdf_with_mineru(self, pdf_url):
        """Use the MinerU API to parse a PDF file  
Returns: str (extracted Markdown text) or None (on failure)"""
        if not MINERU_API_TOKEN:
            log_to_file(f"[MinerU] API Token not configured", "WARNING")
            return None

        log_to_file(f"[MinerU] Starting API-based PDF parsing: {pdf_url}")

        try:
            # Create a parsing task
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MINERU_API_TOKEN}"
            }

            payload = {
                "url": pdf_url,
                "model_version": MINERU_MODEL_VERSION,
                "page_ranges": "1-15"  # Only parse the first 15 pages (Abstract/Methods/Results).
                # Markdown and JSON are the default formats; there is no need to set `extra_formats`.
            }

            log_to_file(f"[MinerU] Creating extraction task...")
            response = requests.post(
                MINERU_API_BASE, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                log_to_file(
                    f"[MinerU] ❌ Task creation failed: HTTP {response.status_code}", "ERROR")
                log_to_file(
                    f"[MinerU]   Response: {response.text[:200]}", "ERROR")
                # Try to parse the error message
                try:
                    error_data = response.json()
                    if 'msg' in error_data:
                        log_to_file(
                            f"[MinerU]   Error message: {error_data['msg']}", "ERROR")
                except:
                    pass
                return None

            result = response.json()
            if result.get("code") != 0:
                log_to_file(
                    f"[MinerU] ❌ API error: {result.get('msg')}", "ERROR")
                return None

            task_id = result["data"]["task_id"]
            log_to_file(f"[MinerU] ✅ Task created: {task_id}")

            # Poll task status
            query_url = f"{MINERU_API_BASE}/{task_id}"
            start_time = time.time()

            while time.time() - start_time < MINERU_MAX_POLL_TIME:
                log_to_file(f"[MinerU] Polling task status...")
                poll_response = requests.get(
                    query_url, headers=headers, timeout=30)

                if poll_response.status_code != 200:
                    log_to_file(
                        f"[MinerU] ⚠️ Poll failed: HTTP {poll_response.status_code}", "WARNING")
                    time.sleep(MINERU_POLL_INTERVAL)
                    continue

                poll_result = poll_response.json()
                if poll_result.get("code") != 0:
                    log_to_file(
                        f"[MinerU] ❌ Poll error: {poll_result.get('msg')}", "ERROR")
                    return None

                state = poll_result["data"]["state"]
                log_to_file(f"[MinerU] Task state: {state}")

                if state == "done":
                    # Task completed. Please download the results.
                    zip_url = poll_result["data"]["full_zip_url"]
                    log_to_file(
                        f"[MinerU] ✅ Task completed, downloading result...")
                    return self._download_and_extract_mineru_result(zip_url)

                elif state == "failed":
                    err_msg = poll_result["data"].get(
                        "err_msg", "Unknown error")
                    log_to_file(f"[MinerU] ❌ Task failed: {err_msg}", "ERROR")
                    return None

                elif state == "running":
                    progress = poll_result["data"].get("extract_progress", {})
                    extracted = progress.get("extracted_pages", 0)
                    total = progress.get("total_pages", 0)
                    log_to_file(
                        f"[MinerU] Processing: {extracted}/{total} pages")

                time.sleep(MINERU_POLL_INTERVAL)

            log_to_file(
                f"[MinerU] ❌ Timeout after {MINERU_MAX_POLL_TIME}s", "ERROR")
            return None

        except requests.exceptions.Timeout:
            log_to_file(f"[MinerU] ❌ Request timeout", "ERROR")
            return None
        except requests.exceptions.RequestException as e:
            log_to_file(
                f"[MinerU] ❌ Request error: {type(e).__name__}: {str(e)}", "ERROR")
            return None
        except Exception as e:
            log_to_file(
                f"[MinerU] ❌ Unexpected error: {type(e).__name__}: {str(e)}", "ERROR")
            return None

    def _download_and_extract_mineru_result(self, zip_url):
        """Download and extract MinerU results; extract full.md plus chart/table metadata.  
Returns: str (enhanced Markdown text) or None (on failure)."""
        try:
            log_to_file(
                f"[MinerU] Downloading result zip from: {zip_url[:80]}...")

            # Download the ZIP file
            response = requests.get(zip_url, timeout=60)
            if response.status_code != 200:
                log_to_file(
                    f"[MinerU] ❌ Download failed: HTTP {response.status_code}", "ERROR")
                return None

            log_to_file(f"[MinerU] Downloaded {len(response.content)} bytes")

            # Decompress and read full.md and chart metadata
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "result.zip")

                # Save ZIP file
                with open(zip_path, 'wb') as f:
                    f.write(response.content)

                # Decompress
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    log_to_file(f"[MinerU] Extracted zip contents")

                # Find the `full.md` file.
                md_file = None
                for root, dirs, files in os.walk(temp_dir):
                    if "full.md" in files:
                        md_file = os.path.join(root, "full.md")
                        break

                if not md_file:
                    log_to_file(
                        f"[MinerU] ❌ full.md not found in zip", "ERROR")
                    return None

                # Read Markdown content
                with open(md_file, 'r', encoding='utf-8') as f:
                    main_content = f.read()

                log_to_file(
                    f"[MinerU] ✅ Extracted {len(main_content)} chars from full.md")

                # Enhancement: Extract chart and table metadata
                enhanced_parts = ["=== MAIN CONTENT ===", main_content]

                # Process content_list.json (contains charts, tables, and metadata)
                content_list_file = None
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('_content_list.json'):
                            content_list_file = os.path.join(root, file)
                            break
                    if content_list_file:
                        break

                if content_list_file:
                    try:
                        with open(content_list_file, 'r', encoding='utf-8') as f:
                            content_list = json.load(f)

                        # Extract metadata (title, author, etc.)
                        metadata = self._extract_metadata_from_content_list(
                            content_list)
                        if metadata:
                            enhanced_parts.insert(0, metadata)
                            enhanced_parts.insert(1, "\n")
                            log_to_file(f"[MinerU] ✅ Extracted paper metadata")

                        # Extract the table
                        tables_info = self._extract_tables_from_layout(
                            content_list)
                        if tables_info:
                            enhanced_parts.append("\n=== EXTRACTED TABLES ===")
                            enhanced_parts.append(tables_info)
                            log_to_file(
                                f"[MinerU] ✅ Extracted {tables_info.count('[TABLE')} tables")

                        # Extract chart
                        figures_info = self._extract_figures_from_layout(
                            content_list)
                        if figures_info:
                            enhanced_parts.append(
                                "\n=== EXTRACTED FIGURES ===")
                            enhanced_parts.append(figures_info)
                            log_to_file(
                                f"[MinerU] ✅ Extracted {figures_info.count('[FIGURE')} figures")

                    except Exception as e:
                        log_to_file(
                            f"[MinerU] ⚠️ Failed to parse content_list.json: {e}", "WARNING")

                # Count the images/ directory
                images_dir = None
                for root, dirs, files in os.walk(temp_dir):
                    if "images" in dirs:
                        images_dir = os.path.join(root, "images")
                        break

                if images_dir and os.path.isdir(images_dir):
                    image_files = [f for f in os.listdir(images_dir)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    log_to_file(
                        f"[MinerU] 📊 Found {len(image_files)} image files")

                # Merge all content
                final_content = "\n\n".join(enhanced_parts)
                log_to_file(
                    f"[MinerU] ✅ Enhanced extraction complete: {len(final_content)} chars total")

                return final_content

        except zipfile.BadZipFile:
            log_to_file(f"[MinerU] ❌ Invalid zip file", "ERROR")
            return None
        except Exception as e:
            log_to_file(
                f"[MinerU] ❌ Extraction error: {type(e).__name__}: {str(e)}", "ERROR")
            return None

    def _extract_tables_from_layout(self, content_list):
        """Extract structured table information from content_list.json.

Args:
    content_list: Parsed content_list.json data (array format)

Returns:
    str: Formatted table description text"""
        try:
            tables = []

            # MinerU’s content_list format is an array, and each element contains a type field.
            if isinstance(content_list, list):
                for item in content_list:
                    if isinstance(item, dict) and item.get('type') == 'table':
                        # Extract table information
                        table_info = {
                            'page': item.get('page_idx', 0) + 1,
                            'bbox': item.get('bbox', []),
                            'latex': item.get('latex', ''),  # LaTeX table
                            'html': item.get('html', ''),  # HTML table
                            'img_path': item.get('img_path', '')  # Path to the table image
                        }
                        tables.append(table_info)

            # Formatted output
            if not tables:
                return ""

            result = []
            for i, table in enumerate(tables, 1):
                result.append(f"\n[TABLE {i}] Page {table['page']}")

                # Prefer using LaTeX (ideally in a structured format).
                if table['latex']:
                    latex_preview = table['latex'][:500]
                    result.append(f"LaTeX: {latex_preview}...")
                # Second, use HTML
                elif table['html']:
                    html_preview = table['html'][:500]
                    result.append(f"HTML: {html_preview}...")
                # Finally, annotate the image path
                elif table['img_path']:
                    result.append(f"Image: {table['img_path']}")

            return "\n".join(result)

        except Exception as e:
            log_to_file(f"⚠️ Failed to extract table information:{str(e)}", "WARNING")
            return ""

    def _extract_figures_from_layout(self, layout_data):
        """Extract structured table information from layout.json

Args:
    layout_data: Parsed layout.json data

Returns:
    str: Formatted table description text"""
        try:
            tables = []

            # MinerU’s `layout.json` format may be a list of pages.
            if isinstance(layout_data, list):
                for page_idx, page in enumerate(layout_data):
                    if isinstance(page, dict) and 'layout_dets' in page:
                        for det in page['layout_dets']:
                            if det.get('category_type') == 'table':
                                # Extract the table position and any possible text content.
                                table_info = {
                                    'page': page_idx + 1,
                                    'bbox': det.get('bbox', []),
                                    'text': det.get('text', ''),
                                    'score': det.get('score', 0)
                                }
                                tables.append(table_info)

            # Formatted output
            if not tables:
                return ""

            result = []
            for i, table in enumerate(tables, 1):
                result.append(f"\n[TABLE {i}] Page {table['page']}")
                if table['text']:
                    result.append(f"Content: {table['text'][:500]}...")  # Limit length

            return "\n".join(result)

        except Exception as e:
            log_to_file(f"⚠️ Failed to extract table information:{str(e)}", "WARNING")
            return ""

    def _extract_figures_from_layout(self, content_list):
        """Extract chart position and type information from content_list.json

Args:
    content_list: Parsed content_list.json data (array format)

Returns:
    str: Formatted chart description text"""
        try:
            figures = []

            if isinstance(content_list, list):
                for item in content_list:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        # Extract chart information
                        figure_info = {
                            'page': item.get('page_idx', 0) + 1,
                            'img_path': item.get('img_path', ''),
                            # Figure caption (possibly a list)
                            'caption': item.get('image_caption', []),
                            'footnote': item.get('image_footnote', []),  # Footnote
                            'bbox': item.get('bbox', [])
                        }
                        figures.append(figure_info)

            # Formatted output
            if not figures:
                return ""

            result = []
            for i, fig in enumerate(figures, 1):
                result.append(f"\n[FIGURE {i}] Page {fig['page']}")
                result.append(f"  Image: {fig['img_path']}")

                # Extract figure captions
                if fig['caption']:
                    if isinstance(fig['caption'], list):
                        caption_text = ' '.join([str(c)
                                                 for c in fig['caption']])
                    else:
                        caption_text = str(fig['caption'])

                    if caption_text.strip():
                        caption_preview = caption_text[:300]
                        result.append(f"  Caption: {caption_preview}...")

                # Extract footnotes
                if fig['footnote']:
                    if isinstance(fig['footnote'], list):
                        footnote_text = ' '.join(
                            [str(f) for f in fig['footnote']])
                    else:
                        footnote_text = str(fig['footnote'])

                    if footnote_text.strip():
                        result.append(f"  Footnote: {footnote_text[:200]}...")

            return "\n".join(result)

        except Exception as e:
            log_to_file(f"⚠️ Failed to extract chart information:{str(e)}", "WARNING")
            return ""

    def _extract_metadata_from_content_list(self, content_list):
        """Extract paper metadata (title, authors, abstract, etc.) from content_list.json

Args:
    content_list: Parsed content_list.json data (array)

Returns:
    str: Formatted metadata text"""
        try:
            metadata_parts = ["=== PAPER METADATA ==="]

            # content_list is an array; you need to find the specified element
            if isinstance(content_list, list):
                title = None
                authors = None
                abstract_parts = []

                for item in content_list:
                    if not isinstance(item, dict):
                        continue

                    # Only process the first page’s content (metadata is usually on the first page).
                    if item.get('page_idx', 0) > 0:
                        continue

                    item_type = item.get('type')
                    text = item.get('text', '')
                    text_level = item.get('text_level', 0)

                    # 1. Extract the title (usually the first relatively long paragraph with text_level=1).
                    if item_type == 'text' and text_level == 1 and not title:
                        # Skip common non-title text such as journal names and author information
                        if len(text) > 20 and 'manuscript' not in text.lower() and 'hhs' not in text.lower():
                            title = text

                    # 2. Extract author information (usually includes superscripts or specific formatting)
                    if item_type == 'text' and not authors:
                        # Check author format: contains ^{} superscript or multiple names
                        if ('$^{' in text or 'and' in text) and len(text) > 30 and len(text) < 300:
                            authors = text

                    # Extract the abstract (usually labeled "Abstract")
                    if item_type == 'text' and 'abstract' in text.lower() and len(text) < 50:
                        # The next item may be summary content
                        abstract_parts = []
                    elif abstract_parts is not None and item_type == 'text' and len(text) > 100:
                        abstract_parts.append(text)
                        if len(' '.join(abstract_parts)) > 500:
                            break  # The abstract is already long enough.

                # Formatted output
                if title:
                    metadata_parts.append(f"Title: {title}")

                if authors:
                    # Clean up LaTeX formatting
                    authors_clean = authors.replace(
                        '$^{', '').replace('}$', '').replace('$', '')
                    metadata_parts.append(f"Authors: {authors_clean[:200]}...")

                if abstract_parts:
                    abstract_text = ' '.join(abstract_parts)[:500]
                    metadata_parts.append(f"Abstract: {abstract_text}...")

            return "\n".join(metadata_parts) if len(metadata_parts) > 1 else ""

        except Exception as e:
            log_to_file(f"⚠️ Failed to extract metadata:{str(e)}", "WARNING")
            return ""

    def _table_to_markdown(self, table):
        """Convert a table extracted by pdfplumber into Markdown format.

Args:
    table: The table returned by pdfplumber (2D list)

Returns:
    str: The table in Markdown format"""
        try:
            if not table or len(table) < 2:
                return ""

            # Data cleaning: replace None with an empty string
            clean_table = [[str(cell or "").strip()
                            for cell in row] for row in table]

            # Ensure all rows have the same number of columns.
            max_cols = max(len(row) for row in clean_table)
            normalized_table = [row + [""] *
                                (max_cols - len(row)) for row in clean_table]

            # Use the first line as the header.
            header = normalized_table[0]
            header_row = "| " + " | ".join(header) + " |"
            separator = "| " + " | ".join(["---"] * len(header)) + " |"

            # Treat all other lines as data.
            data_rows = []
            for row in normalized_table[1:]:
                data_rows.append("| " + " | ".join(row) + " |")

            return "\n".join([header_row, separator] + data_rows)

        except Exception as e:
            log_to_file(f"⚠️ Failed to convert table to Markdown:{str(e)}", "WARNING")
            return ""

    def _parse_pdf(self, pdf_url):
        """Parse the PDF file, preferably using the MinerU API; fall back to pdfplumber if it fails.  
Returns: str (extracted text) or None (on failure)."""
        log_to_file(f"[PDF] Starting PDF parsing: {pdf_url}")

        # Strategy 1: If already configured, try using the MinerU API.
        if MINERU_API_TOKEN:
            log_to_file(f"[PDF] Strategy 1: Trying MinerU API...")
            mineru_result = self._parse_pdf_with_mineru(pdf_url)
            if mineru_result and len(mineru_result) > 200:
                log_to_file(
                    f"[PDF] ✅ MinerU succeeded: {len(mineru_result)} chars")
                return mineru_result
            else:
                log_to_file(
                    f"[PDF] ⚠️ MinerU failed or returned insufficient content, falling back to pdfplumber")

        # Strategy 2: degrade to using pdfplumber
        if not PDFPLUMBER_AVAILABLE:
            log_to_file(f"[PDF] ❌ No PDF parsing method available", "ERROR")
            return None

        log_to_file(f"[PDF] Strategy 2: Trying pdfplumber fallback...")

        try:
            # Download PDF content (using session and full browser request headers)
            log_to_file(f"[pdfplumber] Downloading PDF...")

            # Create a Session to support cookies and redirects.
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            # Add a special Referer for bioRxiv/medRxiv (required!)
            if 'biorxiv.org' in pdf_url or 'medrxiv.org' in pdf_url:
                # Construct the article page URL as the Referer
                article_url = pdf_url.replace('.full.pdf', '').replace('.pdf', '')
                headers['Referer'] = article_url
                log_to_file(f"[pdfplumber] Added bioRxiv Referer: {article_url[:80]}")

            # Visit the homepage first to obtain cookies (for sites that require a session)
            try:
                domain = re.search(r'https?://([^/]+)', pdf_url).group(1)
                session.get(f"https://{domain}", timeout=5, headers=headers)
                log_to_file(
                    f"[pdfplumber] Pre-fetched session cookies from {domain}")
            except:
                pass

            response = session.get(pdf_url, timeout=30,
                                   headers=headers, allow_redirects=True)

            if response.status_code == 403:
                log_to_file(
                    f"[pdfplumber] ⚠️ 403 Forbidden - Access denied by server", "WARNING")
                log_to_file(
                    f"[pdfplumber] This may be a paywall or authentication required")
                return None
            elif response.status_code != 200:
                log_to_file(
                    f"[PDF] ❌ Download failed with status {response.status_code}", "ERROR")
                return None

            # Check the Content-Type and size
            content_type = response.headers.get('Content-Type', '')
            content_length = response.headers.get('Content-Length', 'unknown')

            # Check the file size (to avoid downloading an overly large PDF)
            if content_length != 'unknown':
                try:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > 50:  # Limit to 50 MB
                        log_to_file(
                            f"[pdfplumber] ⚠️ PDF too large: {size_mb:.1f}MB (limit: 50MB)", "WARNING")
                        return None
                except ValueError:
                    pass

            if 'pdf' not in content_type.lower():
                log_to_file(
                    f"[pdfplumber] ⚠️ Content-Type is '{content_type}', may not be a valid PDF")

            log_to_file(
                f"[pdfplumber] Downloaded {len(response.content)} bytes (Content-Length: {content_length})")

            # Parse using pdfplumber (supports table extraction)
            pdf_bytes = BytesIO(response.content)
            extracted_text = []
            page_count = 0
            tables_found = 0

            with pdfplumber.open(pdf_bytes) as pdf:
                page_count = len(pdf.pages)
                log_to_file(f"[pdfplumber] Processing {page_count} pages...")

                # Limit the number of pages to parse (to avoid excessively large PDFs).
                max_pages = min(50, page_count)  # Parse at most the first 50 pages of content.

                for i, page in enumerate(pdf.pages[:max_pages]):
                    try:
                        page_content = []

                        # 1. Extract tables (pdfplumber’s strong suit)
                        tables = page.extract_tables()
                        if tables:
                            for j, table in enumerate(tables):
                                md_table = self._table_to_markdown(table)
                                if md_table:
                                    page_content.append(
                                        f"\n[TABLE Page {i + 1}.{j + 1}]\n{md_table}\n")
                                    tables_found += 1

                        # 2. Extract text
                        text = page.extract_text()
                        if text:
                            page_content.append(text)

                        if page_content:
                            extracted_text.append("\n".join(page_content))

                            # Only record the first page and the last page.
                            if i == 0:
                                log_to_file(
                                    f"[pdfplumber] ✓ Page 1: extracted {len(''.join(page_content))} chars")
                            elif i == max_pages - 1:
                                log_to_file(
                                    f"[pdfplumber] ✓ Page {i + 1}: extracted {len(''.join(page_content))} chars")
                    except Exception as e:
                        log_to_file(
                            f"[pdfplumber] ⚠️ Failed to extract page {i + 1}: {e}")
                        continue

            if tables_found > 0:
                log_to_file(f"[pdfplumber] 📊 Extracted {tables_found} tables")

            if not extracted_text:
                log_to_file(
                    f"[pdfplumber] ❌ No text extracted from PDF", "WARNING")
                return None

            full_text = '\n\n'.join(extracted_text)

            # Clean text (remove extra whitespace)
            import re
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r' {2,}', ' ', full_text)

            log_to_file(
                f"[pdfplumber] ✅ Successfully extracted {len(full_text)} chars from {len(extracted_text)} pages")
            return full_text

        except requests.exceptions.Timeout:
            log_to_file(f"[pdfplumber] ❌ Download timeout: {pdf_url}", "ERROR")
            return None
        except requests.exceptions.RequestException as e:
            log_to_file(
                f"[pdfplumber] ❌ Download error: {type(e).__name__}: {str(e)}", "ERROR")
            return None
        except Exception as e:
            log_to_file(
                f"[pdfplumber] ❌ Parsing error: {type(e).__name__}: {str(e)}", "ERROR")
            return None

    def _extract_pmid_from_url(self, url):
        """Extract PMID from a PubMed URL (supports converting a PMC ID to a PMID)"""
        # 1. Try extracting the PMID directly
        # Optimization: merged patterns and more loosely matched the URL structure
        pmid_match = re.search(
            r'(?:pubmed\.ncbi\.nlm\.nih\.gov|/pubmed)/(\d+)', url)
        if pmid_match:
            return pmid_match.group(1)

        # Detect PMC IDs (compatible with both the legacy www.ncbi.../pmc and the newer pmc.ncbi... formats)
        # Capture group 1 is numeric only.
        pmc_match = re.search(
            r'(?:pmc|ncbi)\.nlm\.nih\.gov/.*articles/PMC(\d+)', url)

        if pmc_match and BIOPYTHON_AVAILABLE:
            # Note: Only the numeric ID is used for the API query here; the “PMC” prefix is not included.
            pmc_uid = pmc_match.group(1)
            full_pmc_string = f"PMC{pmc_uid}"  # For log display only

            log_to_file(
                f"[PubMed] Detected PMC ID: {full_pmc_string}, converting to PMID...")

            try:
                # Rate limiting
                if hasattr(self, 'entrez_delay') and self.entrez_delay:
                    time.sleep(self.entrez_delay)

                # Core fix
                # When dbfrom="pmc", the id must be numeric only (e.g., "9279849").
                handle = Entrez.elink(
                    dbfrom="pmc", db="pubmed", id=pmc_uid, linkname="pmc_pubmed")
                record = Entrez.read(handle)
                handle.close()

                # Parsing result
                # Structural check: ensure LinkSetDb and Link already exist
                if (record and record[0].get('LinkSetDb') and
                        len(record[0]['LinkSetDb']) > 0 and
                        record[0]['LinkSetDb'][0].get('Link')):

                    pmid = record[0]['LinkSetDb'][0]['Link'][0]['Id']
                    log_to_file(
                        f"[PubMed] ✅ Converted {full_pmc_string} -> PMID {pmid}")
                    return pmid
                else:
                    log_to_file(
                        f"[PubMed] ⚠️ No PMID link found for {full_pmc_string}", "WARNING")

            except Exception as e:
                log_to_file(
                    f"[PubMed] ❌ PMC->PMID conversion failed: {e}", "ERROR")

        return None

    def _extract_nct_id_from_url(self, url):
        """Extract the NCT ID from a ClinicalTrials.gov URL  
Returns: NCT ID (str) or None  

Examples:  
- https://clinicaltrials.gov/study/NCT03785249 -> NCT03785249  
- https://clinicaltrials.gov/ct2/show/NCT03785249 -> NCT03785249"""
        # Match strings that start with "NCT" followed by 8 digits
        nct_match = re.search(r'(NCT\d{8})', url, re.IGNORECASE)
        if nct_match:
            nct_id = nct_match.group(1).upper()
            log_to_file(f"[ClinicalTrials] Detected NCT ID: {nct_id} from URL: {url}")
            return nct_id
        return None

    def _fetch_clinicaltrials_study(self, nct_id):
        """Use the ClinicalTrials.gov API v2 to retrieve clinical trial details  
Returns: formatted text summary or None  

API documentation: https://clinicaltrials.gov/api/v2/"""
        log_to_file(f"[ClinicalTrials] Fetching study: {nct_id}")

        base_url = "https://clinicaltrials.gov/api/v2"
        endpoint = f"/studies/{nct_id}"
        full_url = f"{base_url}{endpoint}"

        try:
            response = requests.get(full_url, timeout=15)
            response.raise_for_status()

            data = response.json()
            protocol = data.get('protocolSection', {})

            # Extract the core module
            ident_module = protocol.get('identificationModule', {})
            desc_module = protocol.get('descriptionModule', {})
            cond_module = protocol.get('conditionsModule', {})
            design_module = protocol.get('designModule', {})
            status_module = protocol.get('statusModule', {})
            arms_module = protocol.get('armsInterventionsModule', {})
            eligibility_module = protocol.get('eligibilityModule', {})

            # Build structured text
            formatted_text = f"""Clinical Trial: {nct_id}

Official Title: {ident_module.get('officialTitle', 'N/A')}

Brief Title: {ident_module.get('briefTitle', 'N/A')}

Status: {status_module.get('overallStatus', 'N/A')}

Last Updated: {status_module.get('lastUpdateSubmitDate', 'N/A')}

Conditions: {', '.join(cond_module.get('conditions', ['N/A']))}

Study Type: {design_module.get('studyType', 'N/A')}

Phase: {', '.join(design_module.get('phases', ['N/A']))}

Brief Summary:
{desc_module.get('briefSummary', 'N/A')}

Detailed Description:
{desc_module.get('detailedDescription', 'N/A')}

Interventions:
"""
            # Add detailed information on intervention measures
            interventions = arms_module.get('interventions', [])
            if interventions:
                for i, intervention in enumerate(interventions, 1):
                    formatted_text += f"\n{i}. {intervention.get('type', 'N/A')}: {intervention.get('name', 'N/A')}"
                    if intervention.get('description'):
                        formatted_text += f"\n   Description: {intervention['description']}"
            else:
                formatted_text += "\nNo interventions listed."

            # Add inclusion criteria
            formatted_text += f"\n\nEligibility Criteria:\n{eligibility_module.get('eligibilityCriteria', 'N/A')}"

            formatted_text += f"\n\n[Source: ClinicalTrials.gov API v2]"

            log_to_file(f"[ClinicalTrials] ✅ Successfully fetched: {len(formatted_text)} chars")
            return formatted_text

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 'Unknown'
            log_to_file(f"[ClinicalTrials] ❌ HTTP Error {status_code}: {nct_id}", "ERROR")
            return None
        except requests.exceptions.Timeout:
            log_to_file(f"[ClinicalTrials] ❌ Timeout: {nct_id}", "ERROR")
            return None
        except Exception as e:
            log_to_file(f"[ClinicalTrials] ❌ Error: {type(e).__name__}: {str(e)}", "ERROR")
            return None

    def _search_pubmed_by_title(self, title):
        """Search PubMed for a PMID by article title
Returns: PMID (str) or None"""
        if not BIOPYTHON_AVAILABLE:
            log_to_file(
                "[PubMed Search] ⚠️ Biopython not available", "WARNING")
            return None

        log_to_file(f"[PubMed Search] Searching by title: {title[:100]}...")

        try:
            # Clean the title: remove special characters and keep the core keywords
            cleaned_title = re.sub(r'[^\w\s-]', ' ', title)
            # Only extract the first 150 characters to avoid being too long.
            search_query = cleaned_title[:150].strip()

            # Search using ESearch
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=3,  # Only take the first three results.
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()

            id_list = record.get("IdList", [])
            if id_list:
                pmid = id_list[0]  # Take the first result with the highest relevance.
                log_to_file(f"[PubMed Search] ✅ Found PMID: {pmid}")
                return pmid
            else:
                log_to_file(f"[PubMed Search] ❌ No results found")
                return None

        except Exception as e:
            log_to_file(
                f"[PubMed Search] ❌ Search failed: {type(e).__name__}: {str(e)}", "ERROR")
            return None

    def _fetch_pubmed_abstract(self, pmid):
        """Fetch PubMed abstracts using the Biopython Entrez API  
Returns: dict with keys: title, abstract, authors, journal, date, pmid"""
        if not BIOPYTHON_AVAILABLE:
            log_to_file(
                f"[PubMed] Biopython unavailable, cannot fetch PMID {pmid}", "WARNING")
            return None

        log_to_file(f"[PubMed] Fetching abstract for PMID: {pmid}")

        try:
            # Rate limiting
            if self.entrez_delay:
                import time
                log_to_file(
                    f"[PubMed] Applying rate limit: {self.entrez_delay}s delay")
                time.sleep(self.entrez_delay)

            # Fetch article in XML format
            handle = Entrez.efetch(db="pubmed", id=pmid,
                                   rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            if not records or 'PubmedArticle' not in records or len(records['PubmedArticle']) == 0:
                log_to_file(
                    f"[PubMed] No article found for PMID {pmid}", "WARNING")
                return None

            article = records['PubmedArticle'][0]['MedlineCitation']['Article']

            # Extract Title
            title = article.get('ArticleTitle', 'No Title')

            # Extract Abstract (may have multiple sections)
            abstract_parts = []
            if 'Abstract' in article:
                abstract_texts = article['Abstract'].get('AbstractText', [])
                if isinstance(abstract_texts, list):
                    for section in abstract_texts:
                        if hasattr(section, 'attributes') and 'Label' in section.attributes:
                            label = section.attributes['Label']
                            abstract_parts.append(f"{label}: {str(section)}")
                        else:
                            abstract_parts.append(str(section))
                else:
                    abstract_parts.append(str(abstract_texts))
            abstract = ' '.join(
                abstract_parts) if abstract_parts else 'No abstract available'

            # Extract Authors
            authors = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    if 'LastName' in author and 'Initials' in author:
                        authors.append(
                            f"{author['LastName']} {author['Initials']}")
                    elif 'CollectiveName' in author:
                        authors.append(author['CollectiveName'])
            authors_str = ', '.join(authors[:5])  # Only take the first 5 authors.
            if len(article.get('AuthorList', [])) > 5:
                authors_str += ' et al.'

            # Extract Journal
            journal = article.get('Journal', {}).get(
                'Title', 'Unknown Journal')

            # Extract Date
            pub_date = article.get('Journal', {}).get(
                'JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '')
            day = pub_date.get('Day', '')
            date_str = f"{year} {month} {day}".strip(
            ) if year else 'Unknown Date'

            result = {
                'title': title,
                'abstract': abstract,
                'authors': authors_str,
                'journal': journal,
                'date': date_str,
                'pmid': pmid
            }

            log_to_file(f"[PubMed] ✅ Successfully fetched PMID {pmid}")
            log_to_file(f"[PubMed]   Title: {title[:80]}...")
            log_to_file(f"[PubMed]   Authors: {authors_str}")
            log_to_file(f"[PubMed]   Journal: {journal}")
            log_to_file(f"[PubMed]   Date: {date_str}")
            log_to_file(f"[PubMed]   Abstract length: {len(abstract)} chars")
            return result

        except Exception as e:
            log_to_file(f"[PubMed] Failed to fetch PMID {pmid}: {e}", "ERROR")
            return None

    def _crawl_content(self, url, title=None):
        """Crawl the web page content for the given URL and extract the core text (e.g., abstract).
Enhanced v2: stronger browser header spoofing, smart retries, multi-level fallbacks, avoid nested thread-pool blocking

Args:
    url: target URL
    title: article title (optional, for PubMed fallback search)"""
        try:
            # Priority 1: detect ClinicalTrials.gov links and use the API
            nct_id = self._extract_nct_id_from_url(url)
            if nct_id:
                log_to_file(f"[Crawl] 🏥 Detected ClinicalTrials link: {url}")
                ct_data = self._fetch_clinicaltrials_study(nct_id)
                if ct_data:
                    log_to_file(f"[Crawl] ✅ ClinicalTrials API success: {len(ct_data)} chars")
                    return ct_data
                else:
                    log_to_file(f"[Crawl] ⚠️ ClinicalTrials API failed, falling back to web scraping")

            # Priority 1.5: Detect bioRxiv/medRxiv links (including PDF and web page versions)
            biorxiv_doi = self._extract_biorxiv_doi(url)
            if biorxiv_doi:
                log_to_file(f"[Crawl] 🧬 Detected bioRxiv/medRxiv: {url} -> DOI {biorxiv_doi}")
                # Try to retrieve content via DOI (prefer the PubMed-indexed version).
                pmid = self._search_pubmed_by_doi(biorxiv_doi)
                if pmid:
                    log_to_file(f"[Crawl] ✅ bioRxiv found in PubMed: PMID {pmid}")
                    abstract_data = self._fetch_pubmed_abstract(pmid)
                    if abstract_data:
                        return self._format_pubmed_abstract(abstract_data, "bioRxiv->PubMed")

                # Not available on PubMed; try using the bioRxiv API.
                log_to_file(f"[Crawl] Trying bioRxiv API...")
                biorxiv_data = self._fetch_biorxiv_abstract(biorxiv_doi)
                if biorxiv_data:
                    log_to_file(f"[Crawl] ✅ bioRxiv API success: {len(biorxiv_data)} chars")
                    return biorxiv_data

                # When the API fails, if it’s a PDF link, try downloading it.
                if '.pdf' in url.lower():
                    log_to_file(f"[Crawl] API failed, will try PDF download with special headers...")
                    # Continue executing the PDF processing logic.
                else:
                    log_to_file(f"[Crawl] ⚠️ bioRxiv API failed and not a PDF, skipping")
                    return None

            # Priority 2: Detect PDF links
            if url.lower().endswith('.pdf') or 'pdf' in url.lower():
                log_to_file(f"[Crawl] Detected PDF link: {url}")
                if PDFPLUMBER_AVAILABLE:
                    log_to_file(f"[Crawl] 📄 Attempting PDF parsing...")
                    pdf_text = self._parse_pdf(url)
                    if pdf_text and len(pdf_text) > 200:
                        log_to_file(f"[Crawl] ✅ PDF parsing successful: {len(pdf_text)} chars")
                        return pdf_text
                    else:
                        log_to_file(f"[Crawl] ⚠️ PDF parsing failed or content too short")
                        return None
                else:
                    log_to_file(f"[Crawl] ⚠️ PDF parsing not available (pdfplumber not installed)")
                    return None

            # Priority 3: Enhanced web scraping
            # 1. Enhance disguised browser request headers (simulate a real Chrome 124 browser)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
                'Connection': 'keep-alive'
            }

            log_to_file(f"[Crawl] Fetching URL with browser disguise: {url[:100]}...")

            # 2. Intelligent retry mechanism (to avoid occasional network jitter)
            max_retries = 2
            response = None

            for attempt in range(max_retries):
                try:
                    # verify=False: Skip SSL certificate verification (useful in some corporate firewall environments)
                    # Reduce the timeout to 12 seconds to avoid blocking the outer thread pool.
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=12,
                        verify=False,
                        allow_redirects=True  # Follow the redirect
                    )
                    break  # Stop retrying on success.
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        log_to_file(f"[Crawl] ⏱️ Timeout attempt {attempt + 1}/{max_retries}, retrying...")
                        time.sleep(1)  # Retry after a short delay.
                    else:
                        raise  # Throw an exception if the final retry fails.
                except requests.exceptions.ConnectionError as e:
                    if attempt < max_retries - 1:
                        log_to_file(f"[Crawl] 🔌 Connection error attempt {attempt + 1}/{max_retries}: {str(e)[:100]}")
                        time.sleep(1)
                    else:
                        raise

            if response is None:
                log_to_file(f"[Crawl] ❌ All retry attempts failed")
                return self._fallback_to_pubmed_enhanced(title, url)

            # 3. Explicitly handle anti-scraping status codes
            if response.status_code == 403:
                log_to_file(f"[Crawl] ⛔ 403 Forbidden (WAF/Anti-bot). Triggering enhanced fallback.")
                return self._fallback_to_pubmed_enhanced(title, url)
            elif response.status_code == 401:
                log_to_file(f"[Crawl] 🔒 401 Unauthorized (Login required). Triggering enhanced fallback.")
                return self._fallback_to_pubmed_enhanced(title, url)
            elif response.status_code == 429:
                log_to_file(f"[Crawl] 🚦 429 Rate Limited. Waiting 3s before fallback...")
                time.sleep(3)
                return self._fallback_to_pubmed_enhanced(title, url)

            response.raise_for_status()  # Handle errors such as 404/500

            # Detect Paywall keywords in the HTML hierarchy
            html_text = response.text
            paywall_indicators = [
                'subscription required', 'access denied', 'login required',
                'purchase access', 'institutional access', 'paywall',
                'this article is part of the', 'subscribe to view'
            ]
            html_lower = html_text.lower()
            for indicator in paywall_indicators:
                if indicator in html_lower:
                    log_to_file(f"[Crawl] ⛔ Paywall keyword detected: '{indicator}'")
                    return self._fallback_to_pubmed_enhanced(title, url)

            # 5. Use trafilatura only for content extraction (no downloading)
            text = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=False,
                url=url  # Pass in the URL to handle relative paths
            )

            # 6. Content Quality Check
            if text and len(text) > 200:
                log_to_file(f"[Crawl] ✅ Success: {len(text)} chars extracted")
                return text
            else:
                log_to_file(
                    f"[Crawl] ⚠️ Extracted content too short ({len(text) if text else 0} chars). Triggering fallback.")
                return self._fallback_to_pubmed_enhanced(title, url)

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 'Unknown'
            log_to_file(f"[Crawl] ❌ HTTP Error {status_code}: {url}", "ERROR")
            return self._fallback_to_pubmed_enhanced(title, url)

        except requests.exceptions.Timeout:
            log_to_file(f"[Crawl] ❌ Timeout after {max_retries} attempts: {url}", "ERROR")
            return self._fallback_to_pubmed_enhanced(title, url)

        except requests.exceptions.RequestException as e:
            log_to_file(f"[Crawl] ❌ Network error: {type(e).__name__}: {str(e)[:200]}", "ERROR")
            return self._fallback_to_pubmed_enhanced(title, url)

        except Exception as e:
            log_to_file(f"[Crawl] ❌ Unexpected error: {type(e).__name__}: {str(e)[:200]}", "ERROR")
            return self._fallback_to_pubmed_enhanced(title, url)

    def _fallback_to_pubmed_enhanced(self, title, url):
        """Enhanced fallback strategy: multi-stage attempts to retrieve PubMed content  
Priority: extract PMID from URL -> exact title search -> fuzzy title search -> DOI search  

Args:  
    title: article title (optional)  
    url: original URL  

Returns:  
    str: PubMed abstract text or None"""
        if not BIOPYTHON_AVAILABLE:
            log_to_file(f"[Fallback] ⚠️ Biopython not available, skipping PubMed fallback")
            return None

        log_to_file(f"[Fallback] 🔄 Starting enhanced PubMed fallback for: {url}")

        pmid = None
        fallback_method = None

        # === Strategy 1: Extract the PMID directly from the URL (most reliable) ===
        pmid = self._extract_pmid_from_url(url)
        if pmid:
            fallback_method = "URL_EXTRACTION"
            log_to_file(f"[Fallback] ✅ Strategy 1: Extracted PMID {pmid} from URL")

        # === Strategy 2: Use an exact title search (second most reliable) ===
        if not pmid and title:
            log_to_file(f"[Fallback] 🔍 Strategy 2: Searching by exact title: '{title[:80]}...'")
            pmid = self._search_pubmed_by_title(title)
            if pmid:
                fallback_method = "TITLE_EXACT_SEARCH"
                log_to_file(f"[Fallback] ✅ Found PMID {pmid} via exact title search")

        # Strategy 3: Fuzzy title search (retry after removing special characters)
        if not pmid and title:
            # Clean the title: remove parentheses, colons, and special characters
            cleaned_title = re.sub(r'[:\[\](){}]', ' ', title)
            cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()

            if cleaned_title != title:  # Avoid repeated searches
                log_to_file(f"[Fallback] 🔍 Strategy 3: Fuzzy search with cleaned title: '{cleaned_title[:80]}...'")
                pmid = self._search_pubmed_by_title(cleaned_title)
                if pmid:
                    fallback_method = "TITLE_FUZZY_SEARCH"
                    log_to_file(f"[Fallback] ✅ Found PMID {pmid} via fuzzy title search")

        # === Strategy 4: Try extracting the DOI from the URL and converting it to a PMID (fallback) ===
        if not pmid:
            doi = self._extract_doi_from_url(url)
            if doi:
                log_to_file(f"[Fallback] 🔍 Strategy 4: Searching by DOI: {doi}")
                pmid = self._search_pubmed_by_doi(doi)
                if pmid:
                    fallback_method = "DOI_SEARCH"
                    log_to_file(f"[Fallback] ✅ Found PMID {pmid} via DOI search")

        # === Strategy 5: Special handling for ResearchGate (images/publication URL) ===
        if not pmid and 'researchgate.net' in url:
            log_to_file(f"[Fallback] ⚠️ Strategy 5: ResearchGate URL detected, trying enhanced title search...")

            if title:
                # Remove ResearchGate-specific title formatting
                clean_title = title

                # Remove common prefixes
                clean_title = re.sub(r'^(The|A|An)\s+', '', clean_title, flags=re.IGNORECASE)

                # 2. Remove descriptive suffixes (“TP53 dependency”, “revealing synthetic lethality”, etc.)
                clean_title = re.sub(r'\s+(is|are|reveals?|shows?)\s+.+$', '', clean_title, flags=re.IGNORECASE)

                # Remove redundant conjunctions
                clean_title = re.sub(r'\s+(via|through|by|in|of|and)\s+', ' ', clean_title, flags=re.IGNORECASE)

                # 4. Truncate overly long titles (keep the first 100 characters)
                if len(clean_title) > 100:
                    clean_title = clean_title[:100]

                clean_title = re.sub(r'\s+', ' ', clean_title).strip()

                log_to_file(f"[Fallback]    Cleaned title: '{clean_title[:80]}...'")
                pmid = self._search_pubmed_by_title(clean_title)

                if pmid:
                    fallback_method = "RESEARCHGATE_TITLE_SEARCH"
                    log_to_file(f"[Fallback] ✅ Found PMID {pmid} via ResearchGate title cleanup")

        # === All strategies failed ===
        if not pmid:
            log_to_file(f"[Fallback] ❌ All fallback strategies failed for: {url}")
            return None

        # Retrieve summary data
        abstract_data = self._fetch_pubmed_abstract(pmid)
        if not abstract_data:
            log_to_file(f"[Fallback] ❌ Could not fetch abstract for PMID: {pmid}")
            return None

        # Format the returned text.
        formatted_text = self._format_pubmed_abstract(abstract_data, f"Fallback-{fallback_method}")
        log_to_file(f"[Fallback] ✅ Successfully retrieved abstract via {fallback_method}: {len(formatted_text)} chars")
        return formatted_text

    def _extract_biorxiv_doi(self, url):
        """Extract the DOI from a bioRxiv/medRxiv URL  
Examples:  
    https://www.biorxiv.org/content/10.1101/845446v1 -> 10.1101/845446 (short format)  
    https://www.biorxiv.org/content/10.1101/2020.01.14.905729v1 -> 10.1101/2020.01.14.905729 (standard format)"""
        # bioRxiv/medRxiv DOI format: 10.1101/YYYY.MM.DD.XXXXXX or 10.1101/XXXXXX (older short format)
        # (?:...) denotes a non-capturing group; ? makes the preceding item optional (occurs 0 or 1 time).
        pattern = r'(10\.1101/(?:\d{4}\.\d{2}\.\d{2}\.)?\d+)'
        match = re.search(pattern, url)
        if match:
            doi = match.group(1)
            log_to_file(f"[bioRxiv] Extracted DOI: {doi} from {url[:80]}")
            return doi
        return None

    def _fetch_biorxiv_abstract(self, doi):
        """Fetch the abstract using the bioRxiv API  
API: https://api.biorxiv.org/details/biorxiv/{doi}"""
        try:
            api_url = f"https://api.biorxiv.org/details/biorxiv/{doi}"
            log_to_file(f"[bioRxiv API] Fetching: {api_url}")

            response = requests.get(api_url, timeout=10)
            if response.status_code != 200:
                log_to_file(f"[bioRxiv API] Failed: {response.status_code}")
                return None

            data = response.json()
            if 'collection' not in data or len(data['collection']) == 0:
                log_to_file(f"[bioRxiv API] No data found for DOI {doi}")
                return None

            paper = data['collection'][0]
            formatted_text = f"""Title: {paper.get('title', 'N/A')}

Authors: {paper.get('authors', 'N/A')}

Category: {paper.get('category', 'N/A')}

Date: {paper.get('date', 'N/A')}

DOI: {paper.get('doi', 'N/A')}

Abstract:
{paper.get('abstract', 'No abstract available.')}

[Source: bioRxiv API]"""

            log_to_file(f"[bioRxiv API] ✅ Success: {len(formatted_text)} chars")
            return formatted_text

        except Exception as e:
            log_to_file(f"[bioRxiv API] Error: {type(e).__name__}: {str(e)}", "ERROR")
            return None

    def _extract_doi_from_url(self, url):
        """Extract the DOI (Digital Object Identifier) from a URL.

Examples:
    https://doi.org/10.1038/s41586-021-03819-2 -> 10.1038/s41586-021-03819-2
    https://www.nature.com/articles/s41586-021-03819-2 -> 10.1038/s41586-021-03819-2"""
        # First check bioRxiv
        biorxiv_doi = self._extract_biorxiv_doi(url)
        if biorxiv_doi:
            return biorxiv_doi

        doi_patterns = [
            r'doi\.org/([\d\.]+/[^\s"<>]+)',  # https://doi.org/10.1234/...
            r'/doi/([\d\.]+/[^\s"<>]+)',  # .../doi/10.1234/...
            r'doi:([\d\.]+/[^\s"<>]+)',  # doi:10.1234/...
            r'/articles/([sd]\d+)',  # Nature format: /articles/s41586...
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, url)
            if match:
                doi = match.group(1)
                # Special format conversion for Nature
                if doi.startswith('s') or doi.startswith('d'):
                    doi = f"10.1038/{doi}"
                log_to_file(f"[DOI Extract] Found DOI: {doi}")
                return doi
        return None

    def _search_pubmed_by_doi(self, doi):
        """Search for PMID on PubMed using the DOI"""
        if not BIOPYTHON_AVAILABLE:
            return None

        try:
            Entrez.email = NCBI_EMAIL
            # Use the DOI as the search term.
            handle = Entrez.esearch(
                db="pubmed",
                term=f"{doi}[DOI]",
                retmax=1
            )
            record = Entrez.read(handle)
            handle.close()

            if record['IdList']:
                pmid = record['IdList'][0]
                log_to_file(f"[PubMed DOI Search] Found PMID {pmid} for DOI {doi}")
                return pmid
            return None
        except Exception as e:
            log_to_file(f"[PubMed DOI Search] Error: {str(e)}", "ERROR")
            return None

    def _format_pubmed_abstract(self, abstract_data, source_method):
        """Format PubMed abstract data"""
        formatted_text = f"""Title: {abstract_data.get('title', 'N/A')}

Authors: {abstract_data.get('authors', 'N/A')}

Journal: {abstract_data.get('journal', 'N/A')}

Date: {abstract_data.get('date', 'N/A')}

PMID: {abstract_data.get('pmid', 'N/A')}

Abstract:
{abstract_data.get('abstract', 'No abstract available.')}

[Source: PubMed - Method: {source_method}]"""
        return formatted_text

    def _fallback_to_pubmed_by_title(self, title, url):
        """Legacy fallback API (for backward compatibility); internally calls the enhanced version."""
        return self._fallback_to_pubmed_enhanced(title, url)

    def _load_data(self):
        documents, metadatas = [], []
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                documents.append(item['text'])
                metadatas.append(item['metadata'])

        if os.path.exists(VECTOR_CACHE):
            doc_embeddings = torch.load(VECTOR_CACHE, map_location=DEVICE)
        else:
            raise FileNotFoundError("Vector cache missing.")
        return documents, metadatas, doc_embeddings

        # Added: intent detection function

    async def detect_intent(self, query):
        """Determine whether the user’s intent is casual chat or scientific research."""
        log_to_file("=" * 80)
        log_to_file(f"[Intent Detection] Start intent detection")
        log_to_file(f"[Intent Detection] Input Query:{query}")

        try:
            prompt = INTENT_PROMPT.format(query=query)
            log_to_file(f"[Intent Detection] Full Prompt:{prompt[:200]}...")

            response = await self.async_client.chat.completions.create(
                model=SOLVER_MODEL_NAME,  # Use intern-s1-mini for a quick check
                messages=[
                    {"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature ensures classification stability.
                max_tokens=1000
            )
            intent = response.choices[0].message.content.strip().upper()
            log_to_file(f"[Intent Detection] Model original output:{intent}")

            # To prevent the model from replying with extra punctuation
            if "SCIENTIFIC" in intent:
                final_intent = "SCIENTIFIC"
            else:
                final_intent = "GENERAL"

            log_to_file(f"[Intent Detection] Final decision:{final_intent}")
            log_to_file("=" * 80 + "\n")
            return final_intent
        except Exception as e:
            log_to_file(f"[Intent Detection] Exception:{e}", "ERROR")
            print(f"Intent detection failed: {e}, defaulting to SCIENTIFIC")
            return "SCIENTIFIC"  # Use RAG as a fallback on failure.

    async def _batch_expand_genes(self, genes: List[str]) -> Dict[str, List[str]]:
        """Batch query MyGene.info to retrieve gene aliases (optimize API call efficiency)

Args:
    genes: List of gene symbols

Returns:
    Dict mapping gene -> list of aliases (including official symbol)"""
        if not genes or not USE_GENE_SYNONYM_EXPANSION or not HTTPX_AVAILABLE:
            return {}

        # Filter cached genes
        genes_to_query = [g for g in genes if g not in self.gene_alias_cache]

        if not genes_to_query:
            # All cache hits
            return {g: self.gene_alias_cache.get(g, [g]) for g in genes}

        result_map = {}
        url = "https://mygene.info/v3/query"

        # MyGene.info supports batch queries via POST.
        payload = {
            "q": genes_to_query,
            "scopes": "symbol,alias",
            "fields": "symbol,alias,ensembl.gene",
            "species": "human"
        }

        # Retry mechanism with exponential backoff
        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=5.0,
                        headers={"User-Agent": "SLAgent/1.0"}
                    )
                    data = response.json()

                    # Parsing result
                    for item in data:
                        if isinstance(item, dict):
                            query_gene = item.get('query', '')
                            official_symbol = item.get('symbol', query_gene)
                            aliases_raw = item.get('alias', [])

                            # Normalize the alias list
                            if isinstance(aliases_raw, list):
                                aliases = [official_symbol] + aliases_raw[:MYGENE_MAX_ALIASES]
                            elif isinstance(aliases_raw, str):
                                aliases = [official_symbol, aliases_raw]
                            else:
                                aliases = [official_symbol]

                            # Deduplicate and filter out overly short aliases
                            aliases = list(dict.fromkeys([a for a in aliases if len(a) > 1]))

                            result_map[query_gene] = aliases
                            # Update cache
                            if len(self.gene_alias_cache) < GENE_ALIAS_CACHE_SIZE:
                                self.gene_alias_cache[query_gene] = aliases

                    log_to_file(
                        f"[Gene Expansion] Expanded {len(genes_to_query)} genes -> {sum(len(v) for v in result_map.values())} total aliases")
                    break  # Exit the retry loop after success.

            except Exception as e:
                wait_time = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5 s, 1 s, 2 s
                log_to_file(f"[Gene Expansion] Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s", "WARNING")
                await asyncio.sleep(wait_time)

        # Merge cached results
        for gene in genes:
            if gene not in result_map:
                result_map[gene] = self.gene_alias_cache.get(gene, [gene])

        return result_map

    async def _generate_hyde_document(self, query: str) -> Optional[str]:
        """Generate hypothetical documents (HyDE - Hypothetical Document Embeddings)

Have the LLM "hallucinate" an ideal paper abstract to enhance semantic retrieval.
Especially suitable for sparse-data scenarios, supplementing missing academic terminology in the user's query.

Returns:
    Hypothetical abstract (or None if generation fails)"""
        if not USE_HYDE:
            return None

        prompt = f"""
You are a computational biologist predicting potential synthetic lethal interactions.
User Query: "{query}"

Task: Write a HYPOTHETICAL scientific abstract that proposes a mechanism for this synthetic lethality based on:
1. Paralogs (e.g., ARID1A/ARID1B logic).
2. Collateral lethality (neighboring gene deletion).
3. Pathway compensation (e.g., DNA damage response).

Do NOT invent fake experimental data. Focus on the biological RATIONALE that would make a paper on this topic relevant.
Keywords to include: "compensatory pathway", "loss-of-function", "isogenic cell line", "viability assay".
"""

        try:
            response = await asyncio.wait_for(
                self.async_client.chat.completions.create(
                    model=SOLVER_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,  # A slightly higher temperature allows some creativity.
                    max_tokens=25600
                ),
                timeout=100.0
            )
            hyde_doc = response.choices[0].message.content.strip()
            log_to_file(f"[HyDE] Generated hypothetical document ({len(hyde_doc)} chars)")
            return hyde_doc
        except Exception as e:
            log_to_file(f"[HyDE] Generation failed: {e}", "WARNING")
            return None

    async def generate_sub_queries(self, original_query):
        """Asynchronously generate subqueries (Enhanced: supports gene alias expansion)"""
        log_to_file("=" * 80)
        log_to_file(f"[Query Generation] Start generating sub-queries")
        log_to_file(f"[Query Generation] Original Query:{original_query}")

        # [Added] 1. Extract gene entities and expand their aliases
        entities = self._analyze_query_entities(original_query)
        genes = list(entities['genes'])

        alias_expanded_queries = []
        if genes and USE_GENE_SYNONYM_EXPANSION:
            gene_alias_map = await self._batch_expand_genes(genes)
            if gene_alias_map:
                # Generate an OR query including all aliases
                all_aliases = []
                for gene in genes:
                    aliases = gene_alias_map.get(gene, [gene])
                    all_aliases.extend(aliases[:3])  # Each gene has at most 3 aliases.

                if all_aliases:
                    # Construct a Boolean query: (GENE1 OR ALIAS1 OR …) AND "synthetic lethal"
                    alias_group = " OR ".join([f'"{a}"' for a in all_aliases])
                    expanded_query = f'({alias_group}) AND ("synthetic lethality" OR "synthetic lethal")'
                    alias_expanded_queries.append(expanded_query)
                    log_to_file(f"[Query Gen] Added synonym-expanded query: {expanded_query[:100]}...")

        # 2. LLM generates diverse sub-queries
        prompt = QUERY_GEN_PROMPT.format(question=original_query)
        log_to_file(f"[Query Generation] Prompt length:{len(prompt)}Character")

        try:
            response = await self.async_client.chat.completions.create(
                model=SOLVER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            text = response.choices[0].message.content
            log_to_file(f"[Query Generation] Model original output:\n{text}")

            queries = []
            for line in text.split('\n'):
                clean = re.sub(r'^[\d\.\-\s]+', '', line.strip())
                if clean and not clean.startswith(('#', 'Example')):
                    queries.append(clean)

            # Merge: alias lookup and LLM query
            final_queries = alias_expanded_queries + queries[:4]
            final_queries = final_queries[:4]  # The total number does not exceed 4.

            if not final_queries:
                final_queries = [original_query]

            log_to_file(f"[Query Generation] Number of generated subqueries:{len(final_queries)}")
            for i, q in enumerate(final_queries, 1):
                log_to_file(f"Subquery{i}: {q}")
            log_to_file("=" * 80 + "\n")
            return final_queries
        except Exception as e:
            log_to_file(f"[Query Generation] Exception:{e}", "ERROR")
            print(f"Query gen failed: {e}")
            # Return at least the alias query or the original query.
            return alias_expanded_queries if alias_expanded_queries else [original_query]

    async def _generate_step_back_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Generate biology-mechanism-based step-back queries (for Agentic RAG multi-hop reasoning)

When direct retrieval fails, take a step back and think:
1. Which pathway does this gene belong to?
2. What known paralogs are there?
3. What stress response does gene deletion cause?

Args:
    original_query: Original user query
    num_queries: Number of mechanism-based queries to generate

Returns:
    List of mechanism-based queries"""
        log_to_file("=" * 80)
        log_to_file(f"[Agentic RAG] Step-back reasoning initiated")
        log_to_file(f"[Agentic RAG] Original query: {original_query}")

        prompt = STEP_BACK_PROMPT.format(
            query=original_query,
            num_queries=num_queries
        )

        try:
            response = await self.async_client.chat.completions.create(
                model=SOLVER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5  # Use a slightly higher temperature to increase diversity.
            )
            text = response.choices[0].message.content
            log_to_file(f"[Agentic RAG] Step-back LLM output:\n{text}")

            # Parse query
            mechanistic_queries = []
            for line in text.split('\n'):
                clean = re.sub(r'^[\d\.\-\s]+', '', line.strip())
                if clean and not clean.startswith(('#', 'Example', 'Output')):
                    mechanistic_queries.append(clean)

            # Quantity limit
            mechanistic_queries = mechanistic_queries[:num_queries]

            if not mechanistic_queries:
                log_to_file("[Agentic RAG] ⚠️ Step-back failed, using fallback", "WARNING")
                # Fallback strategy: extract gene names and generate generic mechanism queries
                entities = self._analyze_query_entities(original_query)
                genes = list(entities['genes'])
                if genes:
                    gene_name = genes[0]
                    mechanistic_queries = [
                        f"{gene_name} pathway function mechanism",
                        f"{gene_name} paralog redundancy backup",
                        f"{gene_name} loss cellular stress phenotype"
                    ]
                else:
                    mechanistic_queries = [original_query]

            log_to_file(f"[Agentic RAG] Generated {len(mechanistic_queries)} mechanistic queries:")
            for i, q in enumerate(mechanistic_queries, 1):
                log_to_file(f"  Query {i}: {q}")
            log_to_file("=" * 80 + "\n")

            return mechanistic_queries

        except Exception as e:
            log_to_file(f"[Agentic RAG] Step-back generation failed: {e}", "ERROR")
            print(f"  ❌ Step-back query generation failed: {e}")
            return [original_query]

    async def agentic_retrieval_loop(self, query: str, original_results: List[Dict], max_hops: int = 2) -> List[Dict]:
        """Agentic RAG multi-hop retrieval loop

Core logic:
1. Evaluate the quality of the initial retrieval results (whether the top score is below a threshold)
2. If quality is poor, trigger Step-back reasoning to generate mechanistic queries
3. Perform a second retrieval (based on pathways/mechanisms/orthologs)
4. Merge the initial and second-pass results, then rerank
5. Optional: recursively perform multi-hop retrieval (up to max_hops)

Args:
    query: original user query
    original_results: list of first-round retrieval results (each item includes content, metadata, score)
    max_hops: maximum number of reasoning hops

Returns:
    enhanced result list"""
        if not USE_AGENTIC_RAG:
            log_to_file("[Agentic RAG] DISABLED by config, returning original results")
            return original_results

        # Check the quality of the original results
        if not original_results:
            max_score = 0.0
        else:
            max_score = max(r.get('score', 0.0) for r in original_results)

        log_to_file("=" * 80)
        log_to_file(f"[Agentic RAG] Evaluating retrieval quality")
        log_to_file(f"[Agentic RAG] Max score: {max_score:.4f}, Threshold: {AGENTIC_SCORE_THRESHOLD}")

        # If the result is good enough, return directly.
        if max_score >= AGENTIC_SCORE_THRESHOLD:
            log_to_file(
                f"[Agentic RAG] ✅ Direct retrieval SUCCESSFUL (score={max_score:.4f} >= {AGENTIC_SCORE_THRESHOLD})")
            log_to_file("=" * 80 + "\n")
            print(f"  ✅ [Agentic RAG] Direct retrieval sufficient (score={max_score:.4f})")
            return original_results

        # Trigger multi-hop reasoning
        log_to_file(
            f"[Agentic RAG] ⚠️ Direct retrieval INSUFFICIENT (score={max_score:.4f} < {AGENTIC_SCORE_THRESHOLD})")
        log_to_file(f"[Agentic RAG] 🔄 Initiating Logic-based Multi-hop Reasoning (max_hops={max_hops})")
        print(f"  ⚠️ [Agentic RAG] Low confidence detected (max score={max_score:.4f})")
        print(f"  🔄 [Agentic RAG] Initiating mechanistic reasoning...")

        enhanced_results = list(original_results)  # Copy the original result
        current_hop = 1

        while current_hop <= max_hops:
            log_to_file(f"[Agentic RAG] === HOP {current_hop}/{max_hops} ===")
            print(f"  🧠 [Agentic RAG] Hop {current_hop}/{max_hops}: Generating mechanistic queries...")

            # Step 1: Generate mechanistic queries
            mechanistic_queries = await self._generate_step_back_queries(
                query,
                num_queries=AGENTIC_MECHANISTIC_QUERIES
            )

            if not mechanistic_queries:
                log_to_file(f"[Agentic RAG] ⚠️ Hop {current_hop}: No mechanistic queries generated, stopping")
                print(f"  ⚠️ [Agentic RAG] Hop {current_hop}: Query generation failed, stopping")
                break

            # Step 2: Perform the second retrieval (in parallel)
            log_to_file(
                f"[Agentic RAG] Hop {current_hop}: Executing {len(mechanistic_queries)} mechanistic searches...")
            print(
                f"  🔍 [Agentic RAG] Hop {current_hop}: Searching with {len(mechanistic_queries)} mechanistic queries...")

            secondary_results = []
            seen_links = set(r['metadata'].get('link') for r in enhanced_results if r['metadata'].get('link'))

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Local search + web search
                future_local = executor.submit(self._local_search, mechanistic_queries[0], None)
                future_webs = {executor.submit(self._search_web_single, q): q for q in mechanistic_queries}

                # Collect local results
                try:
                    local_res = future_local.result(timeout=30)
                    for item in local_res:
                        secondary_results.append(item)
                    log_to_file(f"[Agentic RAG] Hop {current_hop}: Local search added {len(local_res)} items")
                except Exception as e:
                    log_to_file(f"[Agentic RAG] Hop {current_hop}: Local search failed: {e}", "WARNING")

                # Collect network results (deduplicate)
                for future in concurrent.futures.as_completed(future_webs, timeout=60):
                    try:
                        web_res = future.result(timeout=5)
                        added = 0
                        for item in web_res:
                            link = item['metadata'].get('link')
                            if link and link not in seen_links:
                                seen_links.add(link)
                                secondary_results.append(item)
                                added += 1
                            elif not link:
                                secondary_results.append(item)
                                added += 1
                        log_to_file(f"[Agentic RAG] Hop {current_hop}: Web search added {added} unique items")
                    except Exception as e:
                        log_to_file(f"[Agentic RAG] Hop {current_hop}: Web search failed: {e}", "WARNING")

            log_to_file(f"[Agentic RAG] Hop {current_hop}: Retrieved {len(secondary_results)} secondary items")
            print(f"  📊 [Agentic RAG] Hop {current_hop}: Retrieved {len(secondary_results)} new items")

            if not secondary_results:
                log_to_file(f"[Agentic RAG] Hop {current_hop}: No secondary results, stopping")
                print(f"  ⚠️ [Agentic RAG] Hop {current_hop}: No new results found, stopping")
                break

            # Step 3: Merge and reorder (rerank)
            log_to_file(f"[Agentic RAG] Hop {current_hop}: Merging and re-ranking...")
            print(
                f"  ⚖️ [Agentic RAG] Hop {current_hop}: Re-ranking {len(enhanced_results) + len(secondary_results)} total items...")

            combined = enhanced_results + secondary_results

            # Rescore
            for item in combined:
                if 'score' not in item or item in secondary_results:
                    # Score the new project
                    score = self.reranker.compute_score(
                        query,
                        item["content"],
                        instruction=None,
                        domain="synthetic_lethality"
                    )
                    item["score"] = score

            # Sorting
            combined.sort(key=lambda x: x["score"], reverse=True)
            enhanced_results = combined

            new_max_score = combined[0]["score"] if combined else 0.0
            log_to_file(f"[Agentic RAG] Hop {current_hop}: New max score: {new_max_score:.4f}")
            print(f"  📈 [Agentic RAG] Hop {current_hop}: New max score: {new_max_score:.4f}")

            # Check whether the satisfaction threshold has been reached.
            if new_max_score >= AGENTIC_SCORE_THRESHOLD:
                log_to_file(
                    f"[Agentic RAG] ✅ Hop {current_hop}: Threshold reached, stopping (score={new_max_score:.4f})")
                print(f"  ✅ [Agentic RAG] Hop {current_hop}: Quality threshold reached!")
                break

            current_hop += 1

        # Final summary
        final_count = len(enhanced_results)
        final_max_score = enhanced_results[0]["score"] if enhanced_results else 0.0
        improvement = final_max_score - max_score

        log_to_file(f"[Agentic RAG] === SUMMARY ===")
        log_to_file(f"[Agentic RAG] Original: {len(original_results)} items, max score={max_score:.4f}")
        log_to_file(f"[Agentic RAG] Enhanced: {final_count} items, max score={final_max_score:.4f}")
        log_to_file(
            f"[Agentic RAG] Improvement: +{improvement:.4f} ({improvement / max_score * 100 if max_score > 0 else 0:.1f}%)")
        log_to_file("=" * 80 + "\n")

        print(f"  ✅ [Agentic RAG] Complete: {final_count} items, score improved by +{improvement:.4f}")

        return enhanced_results

    def _count_tokens(self, text):
        """Calculate the number of tokens in the text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_graph_context(self, graph_context: str, max_tokens: int) -> str:
        """Truncate the graph context to fit the token budget while preserving complete triple lines.

Args:
    graph_context: Graph context string
    max_tokens: Maximum allowed number of tokens

Returns:
    Truncated graph context"""
        if not graph_context:
            return ""

        current_tokens = self._count_tokens(graph_context)
        if current_tokens <= max_tokens:
            return graph_context

        # Split line by line, preserving the header description
        lines = graph_context.split('\n')
        header_lines = []
        triplet_lines = []

        for line in lines:
            if '--[' in line and ']--> ' in line:
                triplet_lines.append(line)
            else:
                header_lines.append(line)

        # Keep the header fixed (sticky).
        header = '\n'.join(header_lines)
        header_tokens = self._count_tokens(header)

        # Compute the remaining number of tokens allocated to the triple
        remaining_tokens = max_tokens - header_tokens - 50  # Leave a little margin.

        if remaining_tokens <= 0:
            return ""

        # Add triples one by one from the beginning until the budget is exceeded.
        kept_triplets = []
        current_triplet_tokens = 0

        for triplet in triplet_lines:
            triplet_tokens = self._count_tokens(triplet + '\n')
            if current_triplet_tokens + triplet_tokens <= remaining_tokens:
                kept_triplets.append(triplet)
                current_triplet_tokens += triplet_tokens
            else:
                break

        if not kept_triplets:
            return ""

        truncated = header + '\n' + '\n'.join(kept_triplets)
        log_to_file(f"[GraphRAG] Truncated graph context: {current_tokens} -> {self._count_tokens(truncated)} tokens")

        return truncated

    def _analyze_query_entities(self, query):
        """Analyze entities in the query (fusion rules + NER, consistent with the BM25 tokenization strategy)."""
        entities = {"genes": set(), "keywords": set()}

        # -------------------------------------------------------
        # 1. Rule/regex-based extraction (run first to ensure obvious gene names are matched)
        # -------------------------------------------------------
        # Note:
        # The preceding character must be the start of the line or a non-uppercase letter/digit (compatible with spaces, punctuation, and Chinese characters).
        # Capture group: matches gene names in the standard format (starts with an uppercase letter, followed by 1–9 uppercase letters or digits)
        # (?=$|[^A-Z0-9]): must be followed by end of line or a non-uppercase-letter/digit character
        # This avoids the issue where `\b` fails with contiguous Chinese text (e.g., "STK11 deletion").
        gene_pattern = r'(?:^|[^A-Z0-9])([A-Z][A-Z0-9]{1,9})(?=$|[^A-Z0-9])'

        # Note: after using `findall` here, you need a bit of cleanup, or just iterate using `search`.
        # For simplicity, we match all all-uppercase strings that conform to gene naming rules.
        potential_genes = re.findall(gene_pattern, query.upper())  # Matching becomes more reliable after converting to uppercase.
        for pg in potential_genes:
            # You can add a length check or a blacklist here to avoid matching things like 'AND', 'OR', etc.
            if len(pg) >= 2 and pg not in ["AND", "NOT", "THE", "FOR"]:
                entities["genes"].add(pg)

        # -------------------------------------------------------
        # 2. Entity extraction based on an NER model (attempting to capture context-relevant entities)
        # -------------------------------------------------------
        try:
            ner_results = self.ner_pipeline(query)
            print(f"    🧬 [NER] Detected entities in query:")

            if not ner_results:
                print(f"       (No entities detected by NER model)")

            # Common entity type mapping for biomedical NER models (aligned with BM25 tokenization)
            gene_like_labels = {
                'Gene_or_gene_product',  # Standard label
                'Gene',  # Simplify tags
                'Protein',  # Protein
                'Coreference',  # Some models will reference this marker gene.
                'Diagnostic_procedure',  # Diagnostic program (usually includes genetic testing markers)
            }

            drug_like_labels = {
                'Drug', 'Chemical', 'Medication',
                'Pharmacologic_substance'
            }

            for entity in ner_results:
                word = entity['word'].strip()
                # Remove the "##" prefix generated by the tokenizer (in BERT architecture)
                word = word.replace('##', '')
                group = entity['entity_group']
                score = entity['score']

                # Only print high-confidence entities
                if score > 0.6 and len(word) >= 3:
                    print(f"       - {word} ({group}, conf: {score:.2f})")

                    if group in gene_like_labels:
                        entities["genes"].add(word.upper())  # Convert uniformly to uppercase.
                    elif group in drug_like_labels:
                        # Add drugs/chemicals to the keywords
                        entities["keywords"].add(word)
                    else:
                        # Also add other entity types to the keywords (e.g., diseases, symptoms, etc.).
                        if len(word) >= 3:  # Filter out words that are too short
                            entities["keywords"].add(word)

        except Exception as e:
            print(f"    ⚠️ [NER] Pipeline extraction error: {e}")
            # NER failure does not affect previous regex matching results

        hardcoded_keywords = ["synthetic lethal", "pathway", "inhibitor", "mutation",
                              "expression", "screening", "CRISPR", "RNAi", "knockdown",
                              "cancer", "tumor", "therapy", "drug", "mechanism",
                              "Synthetic lethality", "Missing", "Inhibitor"]  # Add Chinese keywords supplementary.

        for kw in hardcoded_keywords:
            if kw.lower() in query.lower():
                entities["keywords"].add(kw)

        return entities

    def _boost_by_entity_match(self, candidates, query_entities):
        """Weighted based on entity matching"""
        for cand in candidates:
            meta = cand.get("metadata", {})
            boost = 0.0
            title = meta.get("paper_title", "").lower()
            content = cand.get("content", "").lower()

            # The title hits the core concept (very high weight).
            if "collateral lethality" in title or "synthetic lethal" in title:
                boost += 0.3
            chunk_genes = set(meta.get("key_genes", []))
            if query_entities["genes"] & chunk_genes:
                boost += 0.15
            chunk_methods = set(meta.get("key_methods", []))
            if query_entities["keywords"] & chunk_methods:
                boost += 0.05
            if meta.get("is_web"):
                boost += 0.05
            cand["score"] = cand["score"] * (1 + boost)
        return candidates

    def _diversified_retrieval(self, candidates, min_papers=3, max_per_paper=2):
        """Optimized diversity filtering: for scientific papers, enforce fingerprint deduplication using the Title to resolve duplicates caused by the same paper having different URLs across sites (PubMed, Nature, Cell)."""
        qualified_candidates = [
            c for c in candidates if c["score"] >= SCORE_THRESHOLD]
        if not qualified_candidates:
            return []

        groups = defaultdict(list)

        for cand in qualified_candidates:
            meta = cand.get("metadata", {})
            title = meta.get("paper_title", "").strip()
            link = meta.get("link", "")
            source_id = meta.get("source", "")

            # Core change: build the fingerprint (Fingerprint).
            # Logic: if the title is long enough, use the title directly as the sole ID.
            # Only fall back to using the URL or Source ID when the title is very short or missing.

            # Clean title: remove punctuation and convert to lowercase to avoid treating "Title." and "Title" as two different titles
            clean_title = re.sub(r'[^\w\s]', '', title).lower()

            if len(clean_title) > 15:  # Assume a valid paper title must be at least 15 characters long.
                # Take the first 60 characters as the fingerprint (to prevent weird character differences at the end).
                group_key = clean_title[:60]
            elif link:  # If there’s a link (web source), please use the link.
                group_key = link
            elif source_id:  # Local reference: use the source ID (e.g., "8853")
                group_key = f"local_{source_id}"
            else:  # Use a content hash (to avoid having no identifier at all)
                import hashlib
                content_hash = hashlib.md5(
                    cand["content"][:200].encode()).hexdigest()[:16]
                group_key = f"hash_{content_hash}"

            groups[group_key].append(cand)

        # 3. Polling selection
        selected = []
        group_keys = list(groups.keys())

        # Record the number selected in each group
        selection_counts = defaultdict(int)

        round_index = 0
        MAX_TOTAL_RESULTS = 15  # Maximum total count limit

        while len(selected) < MAX_TOTAL_RESULTS and any(groups.values()):
            # Poll to retrieve the keys of the current group
            current_key = group_keys[round_index % len(group_keys)]

            # Get the remaining candidate items in this group
            current_group_list = groups[current_key]

            if current_group_list:
                # Check whether this group has reached the per-item limit.
                if selection_counts[current_key] < max_per_paper:
                    # Take the highest score in this group (the list is unsorted by default; sort first if recommended).
                    # To improve efficiency, assume the input has been pre-sorted; otherwise, the maximum would need to be found here on the spot.
                    best_cand = max(current_group_list,
                                    key=lambda x: x["score"])

                    selected.append(best_cand)

                    # Remove from the candidate pool and increment the count.
                    current_group_list.remove(best_cand)
                    selection_counts[current_key] += 1
                else:
                    # If this group is already full, stop selecting; clear the group directly or skip it.
                    # To ensure algorithm termination, this simulates clearing this group's candidate eligibility in the current polling round.
                    # (Actual operation: do not select; proceed directly to the next round)
                    pass

            round_index += 1

            # Safety break: if the polling count greatly exceeds the number of groups, it indicates all available options have been selected.
            if round_index > len(group_keys) * max_per_paper * 2:
                break

        return selected

    def _test_serper_api(self):
        """Test whether the Serper API is available"""
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": "test", "num": 1})
            headers = {'X-API-KEY': SERPER_API_KEY,
                       'Content-Type': 'application/json'}

            response = requests.post(
                url, headers=headers, data=payload, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'organic' in data:
                    log_to_file(f"[Serper] ✅ API health check passed")
                    return True
                else:
                    log_to_file(
                        f"[Serper] ⚠️ API response format abnormal:{list(data.keys())}", "WARNING")
                    return False
            elif response.status_code == 401:
                log_to_file(f"[Serper] ❌ The API key is invalid or has expired", "ERROR")
                return False
            elif response.status_code == 429:
                log_to_file(f"[Serper] ❌ API quota exhausted or rate-limited", "ERROR")
                return False
            else:
                log_to_file(f"[Serper] ❌ Status code:{response.status_code}", "ERROR")
                return False
        except Exception as e:
            log_to_file(f"[Serper] ❌ Health check failed:{e}", "ERROR")
            return False

    def _search_web_single(self, query, top_k=20):
        """Web search + intelligent crawling (Deep Fetch)"""
        log_to_file(f"=" * 80)
        log_to_file(f"Start Web search: Query='{query[:100]}'")

        url = "https://google.serper.dev/search"
        payload_dict = {
            "q": query,
            "num": top_k,  # Note: Serper may return more content by default; this is mainly used to control API parameters.
        }

        # Smart time filtering (retain the original logic)
        if re.search(r'\b(latest|recent|202[34]|newest)\b', query, re.IGNORECASE):
            payload_dict["tbs"] = "qdr:y"
            log_to_file(f"Enable time filter: qdr:y")

        payload = json.dumps(payload_dict)
        headers = {'X-API-KEY': SERPER_API_KEY,
                   'Content-Type': 'application/json'}

        candidates = []
        try:
            response = requests.post(
                url, headers=headers, data=payload, timeout=10)

            # Log the full response for diagnostics.
            log_to_file(f"Serper API status code:{response.status_code}")

            if response.status_code != 200:
                log_to_file(f"API request failed: status={response.status_code}", "ERROR")
                log_to_file(f"Response content:{response.text[:500]}", "ERROR")
                print(
                    f"  ❌ Web search failed with status {response.status_code}")
                return []

            response_data = response.json()
            results = response_data.get('organic', [])

            # Check API quota or errors
            if 'error' in response_data:
                log_to_file(f"Serper API error:{response_data['error']}", "ERROR")
                print(f"  ❌ Serper API error: {response_data['error']}")
                return []

            # Check search parameters
            search_params = response_data.get('searchParameters', {})
            log_to_file(
                f"Search parameter: q='{search_params.get('q', 'N/A')}', num={search_params.get('num', 'N/A')}")

            log_to_file(f"API return{len(results)}results")

            # If the result is 0, record the possible causes.
            if len(results) == 0:
                log_to_file(f"⚠️ No search results - Possible reasons:", "WARNING")
                log_to_file(f"1. The query syntax is too strict:{query[:100]}", "WARNING")
                log_to_file(f"2. Serper API quota exhausted", "WARNING")
                log_to_file(f"3. This topic indeed has no relevant content.", "WARNING")

            # Record the original API response (first three entries)
            for idx, r in enumerate(results[:3]):
                log_to_file(
                    f"API raw result[{idx + 1}]: title='{r.get('title', '')[:50]}', date='{r.get('date', 'N/A')}', link={r.get('link', '')}")

            # === Intelligent content extraction: prioritize using the ClinicalTrials API and PubMed API; use Deep Fetch as a fallback ===
            # Step 1: Detect special links (ClinicalTrials, PubMed) and extract them using dedicated APIs.
            special_api_data = {}  # {url: {content, date, title, source_type}}
            urls_to_crawl = []  # Store (url, title) tuples

            for i, res in enumerate(results):
                if i >= 3:  # Only process the first three
                    break
                url = res.get('link')
                title = res.get('title', 'No Title')

                # Priority 1: Detect ClinicalTrials.gov links
                nct_id = self._extract_nct_id_from_url(url)
                if nct_id:
                    log_to_file(f"🏥 ClinicalTrials link detected:{url} -> {nct_id}")
                    ct_info = self._fetch_clinicaltrials_study(nct_id)
                    if ct_info:
                        special_api_data[url] = {
                            'content': ct_info,
                            'date': 'N/A',  # ClinicalTrials returns lastUpdateSubmitDate
                            'title': title,
                            'source_type': 'ClinicalTrials.gov API'
                        }
                        log_to_file(f"✅ ClinicalTrials API succeeded:{len(ct_info)}Character")
                        continue
                    else:
                        log_to_file(f"⚠️ ClinicalTrials API failed, falling back to scraping")
                        urls_to_crawl.append((url, title))
                        continue

                # Priority 2: Detect PubMed links
                pmid = self._extract_pmid_from_url(url)
                if pmid and BIOPYTHON_AVAILABLE:
                    log_to_file(f"🧬 PubMed link detected:{url} -> PMID {pmid}")
                    pubmed_info = self._fetch_pubmed_abstract(pmid)
                    if pubmed_info:
                        # Build structured content
                        structured_content = (
                            f"Title: {pubmed_info['title']}\n"
                            f"Authors: {pubmed_info['authors']}\n"
                            f"Journal: {pubmed_info['journal']}\n"
                            f"Date: {pubmed_info['date']}\n"
                            f"PMID: {pubmed_info['pmid']}\n\n"
                            f"Abstract:\n{pubmed_info['abstract']}"
                        )
                        special_api_data[url] = {
                            'content': structured_content,
                            'date': pubmed_info['date'],
                            'title': pubmed_info['title'],
                            'source_type': 'PubMed API'
                        }
                        log_to_file(
                            f"✅ PubMed API success:{len(structured_content)}Character")
                        continue
                    else:
                        # API request failed; fall back to scraping (pass the title to PubMed as a fallback)
                        log_to_file(f"⚠️ PubMed API failed, falling back to scraping")
                        urls_to_crawl.append((url, title))
                        continue

                # Priority 3: other links; use traditional crawling (pass the title for PubMed fallback)
                urls_to_crawl.append((url, title))

            # Step 2: Use Deep Fetch for non-special API links
            # Optimization: avoid conflicts from nested timeouts; each crawler task is limited to 20s max.
            crawled_data = {}
            if urls_to_crawl:
                log_to_file(f"Start Deep Fetch:{len(urls_to_crawl)}URLs", console=True)
                print(f"  🕷️ Deep fetching {len(urls_to_crawl)} URLs...")

                # Use single-threaded sequential crawling to avoid deadlocks from nested thread pools.
                for idx, (url, title) in enumerate(urls_to_crawl, 1):
                    print(f"    [{idx}/{len(urls_to_crawl)}] Crawling: {url[:60]}...")
                    try:
                        # Set a 20-second timeout (12-second requests timeout inside _crawl_content + retries).
                        text = self._crawl_content(url, title)

                        if text:
                            if len(text) > 200:
                                crawled_data[url] = text
                                log_to_file(f"  ✅ Crawl success: {url} -> {len(text)} chars", console=True)
                                print(f"       ✅ Success: {len(text)} chars")
                            else:
                                log_to_file(f"  ⚠️ Content too short: {url} (only {len(text)} chars, min 200)",
                                            console=True)
                                print(f"       ⚠️ Too short: {len(text)} chars")
                        else:
                            log_to_file(f"  ❌ Crawl failed: {url} (returned None)", console=True)
                            print(f"       ❌ Failed (None returned)")

                    except Exception as e:
                        log_to_file(f"  ❌ Crawl exception: {url} -> {type(e).__name__}: {str(e)[:200]}", "ERROR",
                                    console=True)
                        print(f"       ❌ Exception: {type(e).__name__}")

                print(f"  🎯 Deep Fetch completed: {len(crawled_data)}/{len(urls_to_crawl)} successful")
            else:
                # All URLs have been handled by a dedicated API.
                print(f"  ℹ️ All Top-3 URLs handled by special APIs (ClinicalTrials/PubMed), no Deep Fetch needed")

            # Assemble the final result
            for i, res in enumerate(results):
                title = res.get('title', 'No Title')
                link = res.get('link', '')
                snippet = res.get('snippet', '')
                date_original = res.get('date', 'Unknown Date')  # Save the raw date returned by the API
                date = date_original

                log_to_file(
                    f"Processing result {i + 1}: Title='{title[:50]}...', Link={link}, Date_API='{date_original}'")

                # Priority: dedicated APIs (ClinicalTrials/PubMed) > deep crawling > snippets
                if link in special_api_data:
                    # Highest priority: use structured data extracted via dedicated APIs
                    api_data = special_api_data[link]
                    body_content = api_data['content'][:3000]
                    content_source = api_data['source_type']
                    date = api_data['date'] if api_data['date'] != 'N/A' else date_original
                    title = api_data['title']
                    log_to_file(
                        f"✅ Use{content_source}Data, content length={len(body_content)}Characters, date={date}")

                elif link in crawled_data:
                    # Secondary priority: use the content fetched by Deep Fetch
                    body_content = crawled_data[link][:2500]
                    content_source = "Deep Fetch (Full Abstract)"
                    log_to_file(
                        f"  ✅ Deep Fetch success, content length={len(body_content)} chars")

                    # Optimized date extraction strategy
                    if date_original == "Unknown Date" or date_original == "Recent":
                        log_to_file(
                            f"  📅 API date invalid ({date_original}), extracting from content...")

                        # Improved regular expression: matches multiple date formats
                        date_patterns = [
                            # "January 15, 2024"
                            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b',
                            # "15 January 2024"
                            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b',
                            # "2024-01-15" or "2024/01/15"
                            r'\b(20\d{2})[-/](0?\d|1[0-2])[-/](0?\d|[12]\d|3[01])\b',
                            r'Published:\s*(20\d{2})',  # "Published: 2024"
                            r'\b(20\d{2})\b'  # Fallback: any 4-digit year
                        ]

                        extracted_date = None
                        for pattern in date_patterns:
                            matches = re.findall(
                                pattern, body_content, re.IGNORECASE)
                            if matches:
                                if isinstance(matches[0], tuple):
                                    # Complete date format
                                    extracted_date = ' '.join(
                                        str(x) for x in matches[0] if x)
                                else:
                                    extracted_date = matches[0]
                                log_to_file(
                                    f"  📅 Pattern matched: '{pattern[:50]}...' -> {extracted_date}")
                                break

                        if extracted_date:
                            date = extracted_date
                            log_to_file(
                                f"  ✅ Date extracted from content: {date}")
                        else:
                            log_to_file(f"  ⚠️ No valid date found in content")
                    else:
                        log_to_file(f"  📅 Using API date: {date}")
                else:
                    # Lowest priority: use Google code snippet
                    body_content = snippet
                    content_source = "Google Snippet"
                    log_to_file(
                        f"Using Snippet, length={len(body_content)}Characters, date={date}")

                # Format the content for LLM reading.
                content = (
                    f"Title: {title}\n"
                    f"Date: {date}\n"
                    f"Link: {link}\n"
                    f"Content Type: {content_source}\n"
                    f"Body: {body_content}\n"
                )

                candidates.append({
                    "content": content,
                    "metadata": {
                        "source": "web_search",
                        "paper_title": title,
                        "key_genes": [],
                        "is_web": True,
                        "link": link
                    },
                    "score": 0.0  # Initial score, pending re-ranking
                })

                log_to_file(
                    f"Final candidate item[{i + 1}]: Date={date}, Source={content_source}")

            log_to_file(f"Web search completed: returned{len(candidates)}Candidate item")
            log_to_file(f"=" * 80 + "\n")
            return candidates

        except requests.exceptions.Timeout:
            log_to_file(f"Serper API timeout:{query[:50]}", "ERROR")
            print(f"  ❌ Web search timeout for '{query[:20]}...'")
            return []
        except requests.exceptions.RequestException as e:
            log_to_file(
                f"Serper API request error:{type(e).__name__}: {str(e)}", "ERROR")
            print(f"  ❌ Web search error for '{query[:20]}...': {e}")
            return []
        except Exception as e:
            log_to_file(f"Web search error:{type(e).__name__}: {str(e)}", "ERROR")
            print(f"  ❌ Unexpected error in web search: {e}")
            return []

    def _tokenize_for_bm25(self, text):
        """Tokenization for BM25 - Optimized for Biomedical Text (Rule-Based + NER Fusion)"""
        # Strategy: combine regular expressions with an NER model to extract proper nouns and general vocabulary

        final_tokens = []

        # Part 1: Rule-based regex extraction (fast, reliable)
        # 1.1 Extract all words (including hyphenated words)
        all_tokens = re.findall(r'\b[\w-]+\b', text)

        # 1.2 For each token: if it matches an all-uppercase gene name, keep both the original form and the lowercase form.
        gene_pattern = re.compile(r'^[A-Z][A-Z0-9-]{1,10}$')

        for token in all_tokens:
            if gene_pattern.match(token):
                # Gene name: preserve the original capitalization and provide a lowercase version
                final_tokens.append(token)  # Uppercase: BRCA1
                final_tokens.append(token.lower())  # Lowercase: brca1
            else:
                # Common vocabulary: keep only the lowercase form
                final_tokens.append(token.lower())

        # ============ Part 2: NER Model Enhancement (Intelligent Proper Noun Recognition) ============
        # Only use named entity recognition for shorter texts (to avoid performance issues).
        if len(text) < 500 and hasattr(self, 'ner_pipeline'):
            try:
                ner_results = self.ner_pipeline(text)

                # Mapping of common entity types in biomedical NER models
                # Note: Label names may vary across models; this supports compatibility with multiple labeling schemes.
                gene_like_labels = {
                    'Gene_or_gene_product',  # Standard label
                    'Gene',  # Simplify tags
                    'Protein',  # Protein
                    'Coreference',  # Some models will reference this marker gene.
                    'Diagnostic_procedure',  # Diagnostic program (usually includes genetic testing markers)
                }

                drug_like_labels = {
                    'Drug', 'Chemical', 'Medication',
                    'Pharmacologic_substance'
                }

                for entity in ner_results:
                    word = entity['word'].strip().replace(
                        '##', '')  # Clean up BERT tokenizer prefixes
                    group = entity['entity_group']
                    score = entity['score']

                    # Lower the threshold to 0.6 (balancing precision and recall)
                    # Also filter out tokens that are too short (length >= 3) to avoid noise
                    if score > 0.6 and len(word) >= 3:
                        if group in gene_like_labels:
                            # Gene name: include both uppercase and lowercase versions simultaneously
                            final_tokens.append(word.upper())
                            final_tokens.append(word.lower())
                        elif group in drug_like_labels:
                            # Drugs/chemicals: keep as-is and convert to lowercase
                            final_tokens.append(word)
                            final_tokens.append(word.lower())
            except Exception as e:
                # NER failure does not affect basic tokenization
                log_to_file(
                    f"[BM25 Tokenizer] NER extraction failed: {e}", "WARNING")

        return final_tokens

    def _reciprocal_rank_fusion(self, rankings_list, k=60):
        """Use RRF to fuse multiple ranked lists

Args:
    rankings_list: List of lists of [(doc_id, score), ...]
    k: RRF constant, default 60

Returns:
    Fused [(doc_id, fused_score), ...] sorted by score in descending order"""
        fused_scores = {}

        for rankings in rankings_list:
            for rank, (doc_id, _) in enumerate(rankings, start=1):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank)

        # Sort in descending order by the fusion score.
        sorted_docs = sorted(fused_scores.items(),
                             key=lambda x: x[1], reverse=True)
        return sorted_docs

    def _local_search(self, query, hyde_doc: Optional[str] = None):
        """Local retrieval (supports HyDE hybrid retrieval)

Args:
    query: User's original query
    hyde_doc: Hypothetical document generated by HyDE (optional)

Returns:
    List of candidate documents"""
        log_to_file("=" * 80)
        log_to_file(f"[Local Search] Start local search: Query='{query}'")
        if hyde_doc:
            log_to_file(f"[Local Search] HyDE Mode: Use Hypothetical Documents to Assist Retrieval")

        # ============ Part 1A: Raw Query Dense Retrieval ============
        log_to_file(f"[Local Search - Dense-Raw] Raw query vector retrieval...")
        instr = "Given a web search query, retrieve relevant passages that answer the query"
        q_vec_raw = self.embedder.encode(
            [query], is_query=True, task_instruction=instr)

        raw_scores = torch.mm(
            q_vec_raw, self.docs_vecs.transpose(0, 1)).cpu().numpy()[0]

        raw_top_indices = np.argsort(raw_scores)[::-1][:100]
        raw_rankings = [(idx, raw_scores[idx]) for idx in raw_top_indices]

        log_to_file(f"[Local Search - Dense-Raw] Top 5 Similarities:{[f'{s:.4f}' for s in raw_scores[raw_top_indices[:5]]]}")

        # Part 1B: HyDE dense retrieval (optional)
        hyde_rankings = []
        if hyde_doc and USE_HYDE:
            log_to_file(f"[Local Search - Dense-HyDE] HyDE document vector retrieval...")
            q_vec_hyde = self.embedder.encode(
                [hyde_doc], is_query=True, task_instruction=instr)

            hyde_scores = torch.mm(
                q_vec_hyde, self.docs_vecs.transpose(0, 1)).cpu().numpy()[0]

            hyde_top_indices = np.argsort(hyde_scores)[::-1][:100]
            hyde_rankings = [(idx, hyde_scores[idx]) for idx in hyde_top_indices]

            log_to_file(
                f"[Local Search - Dense-HyDE] Top 5 Similarities:{[f'{s:.4f}' for s in hyde_scores[hyde_top_indices[:5]]]}")

        # ============ Part 2: Sparse Retrieval (BM25) ============
        sparse_rankings = []
        if self.bm25 is not None:
            log_to_file(f"[Local Search - Sparse] Using BM25 keyword matching...")
            query_tokens = self._tokenize_for_bm25(query)
            log_to_file(f"[Local Search - Sparse] Query tokens: {query_tokens[:10]}...")

            bm25_scores = self.bm25.get_scores(query_tokens)
            sparse_top_indices = np.argsort(bm25_scores)[::-1][:100]
            sparse_rankings = [(idx, bm25_scores[idx]) for idx in sparse_top_indices]

            log_to_file(
                f"[Local Search - Sparse] Top 5 BM25 scores:{[f'{s:.4f}' for s in bm25_scores[sparse_top_indices[:5]]]}")
        else:
            log_to_file(f"[Local Search - Sparse] BM25 not enabled, skipping")

        # ============ Part 3: Multi-Source Fusion (RRF) ============
        # Collect all leaderboards
        all_rankings = [raw_rankings]
        ranking_labels = ["Dense-Raw"]

        if hyde_rankings:
            all_rankings.append(hyde_rankings)
            ranking_labels.append("Dense-HyDE")

        if sparse_rankings:
            all_rankings.append(sparse_rankings)
            ranking_labels.append("Sparse-BM25")

        log_to_file(f"[Local Search - Fusion] Fusion{len(all_rankings)}Rank:{', '.join(ranking_labels)}")

        if len(all_rankings) > 1:
            fused_results = self._reciprocal_rank_fusion(all_rankings, k=60)
            final_indices = [doc_id for doc_id, _ in fused_results[:40]]
            log_to_file(f"[Local Search - Fusion] Return Top- after RRF fusion{len(final_indices)}Document")
        else:
            # Only the original query; use it directly.
            final_indices = raw_top_indices[:40]
            log_to_file(f"[Local Search - Fusion] Single search source, returns Top-{len(final_indices)}Document")

        log_to_file(f"[Local Search] Final return{len(final_indices)}Candidate documents")
        log_to_file("=" * 80 + "\n")

        return [{"content": self.docs_text[i], "metadata": self.docs_meta[i], "score": 0.0} for i in final_indices]


# ================= FastAPI Setup =================


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\U0001f680 Initializing RAG engine...")
    # Set proxy
    os.environ['http_proxy'] = PROXY_URL
    os.environ['https_proxy'] = PROXY_URL

    global_resources["rag"] = RAGLogic()
    print("\u2705 System ready!")
    yield
    print("\U0001f6d1 Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    rag: RAGLogic = global_resources["rag"]
    user_query = request.query
    client = rag.async_client

    async def event_generator():
        # === [Added] Step 0: Intent Recognition ===
        yield json.dumps({"type": "progress", "data": "\U0001f9e0 Analyzing user intent..."}, ensure_ascii=False) + "\n"

        intent = await rag.detect_intent(user_query)

        # === Branch A: General small talk (reply directly) ===
        if intent == "GENERAL":
            yield json.dumps({"type": "progress", "data": "\U0001f4ac General conversation detected, switching to chat mode..."},
                             ensure_ascii=False) + "\n"

            # 1. Send an empty reference list (to clear any cards that may remain on the frontend)
            yield json.dumps({"type": "references", "data": []}, ensure_ascii=False) + "\n"

            # Call the Solver to generate a response and provide guidance
            messages = [
                {"role": "system", "content": "You are a helpful scientific assistant."},
                {"role": "user", "content": GENERAL_CHAT_PROMPT.format(
                    query=user_query)}
            ]

            response = await client.chat.completions.create(
                model=SOLVER_MODEL_NAME,
                messages=messages,
                temperature=0.3,
                stream=True
            )

            yield json.dumps({"type": "progress", "data": "\u2728 Replying..."}, ensure_ascii=False) + "\n"

            # 3. Streaming output
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield json.dumps({"type": "token", "data": content}, ensure_ascii=False) + "\n"

            # In chit-chat mode, terminate immediately without executing the subsequent RAG pipeline.
            return

        # === Branch B: Research questions (execute RAG) ===
        yield json.dumps({"type": "progress", "data": "\U0001f52c Identified as a scientific research question; starting the RAG engine..."}, ensure_ascii=False) + "\n"

        # 1. Query Generation
        yield json.dumps({"type": "progress", "data": "\U0001f50d [Planner] Generating multi-dimensional search keywords..."},
                         ensure_ascii=False) + "\n"
        sub_queries = await rag.generate_sub_queries(user_query)
        print(f"  🧠 [Planner] Generated {len(sub_queries)} sub-queries:")
        for i, q in enumerate(sub_queries, 1):
            print(f"     {i}. {q}")

        # [Added] 1.5 HyDE generation (asynchronous)
        hyde_doc = None
        if USE_HYDE:
            yield json.dumps({"type": "progress", "data": "\U0001f52e [HyDE] Generate hypothetical documents to enhance retrieval..."},
                             ensure_ascii=False) + "\n"
            hyde_doc = await rag._generate_hyde_document(user_query)
            if hyde_doc:
                print(f"  🔮 [HyDE] Generated hypothetical doc: {hyde_doc[:150]}...")

        # 2. Parallel Search
        yield json.dumps(
            {"type": "progress", "data": f"🌍 [Retriever] Execute parallel search (1 local +{len(sub_queries)}network)..."},
            ensure_ascii=False) + "\n"

        all_candidates = []
        seen_links = set()  # URL deduplication

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # A. Local tasks (using the original query + HyDE document)
            future_local = executor.submit(rag._local_search, user_query, hyde_doc)

            # B. Network tasks (using the generated subqueries)
            future_webs = {executor.submit(
                rag._search_web_single, q): q for q in sub_queries}

            # Collect local results (add timeout protection)
            try:
                local_res = future_local.result(timeout=30)  # 30-second timeout
                all_candidates.extend(local_res)
                print(f"  ✅ [Local] Retrieved {len(local_res)} candidates")
            except concurrent.futures.TimeoutError:
                print(f"  ⏱️ [Local] Search timeout (>30s)")
            except Exception as e:
                print(f"  ❌ [Local] Search failed: {e}")

            # Collect web results (URL deduplication + timeout protection)
            completed = 0
            timeout_count = 0

            # Collect network request results (with timeout protection and error handling)
            completed = 0
            timeout_count = 0

            try:
                for future in concurrent.futures.as_completed(future_webs, timeout=60):  # Overall maximum 60 seconds
                    query = future_webs[future]
                    try:
                        web_res = future.result(timeout=5)  # Maximum duration for a single result is 5 seconds.
                        added = 0
                        for item in web_res:
                            link = item['metadata'].get('link')
                            # URL deduplication
                            if link and link not in seen_links:
                                seen_links.add(link)
                                all_candidates.append(item)
                                added += 1
                            elif not link:  # Local or other unlinked content
                                all_candidates.append(item)
                                added += 1

                        completed += 1
                        print(f"  ✅ [Web '{query[:30]}...'] Retrieved {len(web_res)} items, {added} added after dedup")

                    except concurrent.futures.TimeoutError:
                        timeout_count += 1
                        print(f"  ⏱️ [Web '{query[:30]}...'] Timeout")
                    except Exception as e:
                        print(f"  ❌ [Web '{query[:30]}...'] Failed: {type(e).__name__}: {str(e)[:100]}")

            except concurrent.futures.TimeoutError:
                # The total timeout is 60 seconds.
                unfinished = len(future_webs) - completed - timeout_count
                print(
                    f"  ⚠️ [Web] Overall timeout: {completed} completed, {timeout_count} timed out, {unfinished} unfinished")

        print(
            f"     ∑ [Retriever] Aggregated {len(all_candidates)} candidates.")

        # 3. Reranking
        yield json.dumps({"type": "progress", "data": f"⚖️ [Ranker] Scoring {len(all_candidates)} results for relevance..."},
                         ensure_ascii=False) + "\n"

        if not all_candidates:
            print("  ⚠️ [Ranker] No candidates to rank!")
            yield json.dumps({"type": "references", "data": []}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "progress", "data": "\u26a0\ufe0f No relevant content retrieved; switching to general knowledge Q&A..."},
                             ensure_ascii=False) + "\n"
            # Enter general response mode directly
            prompt = f"User Query: {user_query}\nNo specific papers found. Answer based on general scientific knowledge."
            messages = [
                {"role": "system", "content": "You are a scientific assistant."},
                {"role": "user", "content": prompt}
            ]
            response = await client.chat.completions.create(
                model=SOLVER_MODEL_NAME, messages=messages, temperature=0.7, stream=True
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield json.dumps({"type": "token", "data": chunk.choices[0].delta.content},
                                     ensure_ascii=False) + "\n"
            return

        scored = []
        # Use SL-specific reranker instructions
        for cand in all_candidates:
            s = rag.reranker.compute_score(
                user_query, cand["content"],
                instruction=None,  # Let the reranker automatically choose the instruction based on the domain.
                domain="synthetic_lethality"  # Specify the SL domain
            )
            cand["score"] = s
            scored.append(cand)

        scored.sort(key=lambda x: x["score"], reverse=True)
        print(f"  📊 [Ranker] Top score: {scored[0]['score']:.4f}")

        # [New] Agentic RAG: Enhanced multi-hop reasoning
        if USE_AGENTIC_RAG:
            yield json.dumps({"type": "progress", "data": "\U0001f9e0 [Agentic RAG] Evaluate retrieval quality, and initiate multi-hop reasoning when necessary..."},
                             ensure_ascii=False) + "\n"
            scored = await rag.agentic_retrieval_loop(
                user_query,
                scored,
                max_hops=AGENTIC_MAX_HOPS
            )
            # Resort (agentic_retrieval_loop is already sorted, but just in case)
            scored.sort(key=lambda x: x["score"], reverse=True)
            if scored:
                print(f"  📊 [Agentic RAG] Enhanced score: {scored[0]['score']:.4f}")

        # [Critical] First analyze entities: this is a shared prerequisite dependency for Entity Boost and Graph RAG
        # Perform entity analysis regardless of whether USE_ENTITY_BOOST is enabled (required by Graph RAG).
        print(f"  🧬 [NER] Analyzing query entities...")
        ents = rag._analyze_query_entities(user_query)
        print(f"       Genes: {list(ents['genes'])[:5]}")
        print(f"       Keywords: {list(ents['keywords'])[:5]}")

        # [New] Knowledge base enhancement — inject SL pairs already known to DepMap
        if USE_DEPMAP_KB and ents['genes']:
            print(f"  📚 [KB] Augmenting with DepMap knowledge base...")
            yield json.dumps({"type": "progress", "data": "\U0001f4da [KB] Querying DepMap known SL data..."}, ensure_ascii=False) + "\n"

            synthetic_docs = rag.kb_augmenter.augment_with_depmap(list(ents['genes']))
            if synthetic_docs:
                print(f"  📚 [KB] Added {len(synthetic_docs)} synthetic documents from DepMap")
                # Add synthetic documents to the candidate list and assign them higher scores (to ensure they are considered).
                for doc in synthetic_docs:
                    doc['score'] = 0.95  # High-priority score
                    scored.append(doc)
                print(f"  📚 [KB] Total candidates after KB augmentation: {len(scored)}")

        # Entity enhancement (optional, but recommended to enable to ensure Top-K quality)
        if USE_ENTITY_BOOST:
            scored = rag._boost_by_entity_match(scored, ents)
            scored.sort(key=lambda x: x["score"], reverse=True)
            print(
                f"  ✨ [NER] After boost, top score: {scored[0]['score']:.4f}")

        # Diversity filtering (round-robin)
        if USE_ROUND_ROBIN:
            print(f"  🎯 [Diversity] Applying Round-Robin selection...")
            final_res = rag._diversified_retrieval(
                scored, MIN_PAPERS, MAX_CHUNKS_PER_PAPER)
            print(f"  ✅ [Diversity] Selected {len(final_res)} diverse chunks")
        else:
            print(f"  📋 [Selection] Using simple top-k selection...")
            final_res = scored[:8]

        # Context compression
        if USE_CONTEXT_COMPRESSION and final_res:
            yield json.dumps(
                {"type": "progress", "data": f"🗜️ [Compressor] Compress{len(final_res)}A snippet to extract key information..."},
                ensure_ascii=False) + "\n"
            print(
                f"  🗜️ [Compressor] Compressing {len(final_res)} chunks (Strategy: {COMPRESSION_STRATEGY})...")

            original_total_len = sum(len(r["content"]) for r in final_res)
            final_res = await rag.compressor.compress_chunks(user_query, final_res)
            compressed_total_len = sum(len(r["content"]) for r in final_res)

            compression_ratio = compressed_total_len / \
                                original_total_len if original_total_len > 0 else 1.0
            print(
                f"  ✅ [Compressor] Compression complete: {original_total_len} -> {compressed_total_len} chars ({compression_ratio * 100:.1f}%)")
            yield json.dumps(
                {"type": "progress", "data": f"✅ Compression completed: Kept{compression_ratio * 100:.1f}Key content of %"},
                ensure_ascii=False) + "\n"

        # ========== External Knowledge Graph (P1 Priority - Highest Confidence) ==========
        external_kg_context = ""
        external_kg_tokens = 0

        if USE_EXTERNAL_KG and rag.external_kg and rag.external_kg.available:
            try:
                yield json.dumps(
                    {"type": "progress", "data": "\U0001f52c [External KG] Query external knowledge graphs (DepMap/DrugBank/Bgee)..."},
                    ensure_ascii=False) + "\n"
                print(f"  🔬 [External KG] Querying raw_kg.tsv for entities...")

                # Extract query entities (genes and drugs)
                query_entities = list(ents['genes'])[:5]  # Limit to at most 5 genes
                # Potential drug names can also be extracted from the keywords

                if query_entities:
                    print(f"       Query entities: {query_entities}")

                    # Query an external knowledge graph (1-hop or 2-hop).
                    raw_external_kg = rag.external_kg.query_subgraph(
                        entities=query_entities,
                        hops=EXTERNAL_KG_HOPS
                    )

                    if raw_external_kg:
                        # Calculate the number of tokens
                        raw_kg_tokens = rag._count_tokens(raw_external_kg)

                        # Truncate to the budget
                        if raw_kg_tokens > TOKEN_BUDGET_EXTERNAL_KG_MAX:
                            print(
                                f"  ✂️ [External KG] Truncating: {raw_kg_tokens} -> {TOKEN_BUDGET_EXTERNAL_KG_MAX} tokens")
                            # Simple truncation: keep the first N lines until the budget is reached
                            lines = raw_external_kg.split('\n')
                            truncated_lines = []
                            temp_tokens = 0
                            for line in lines:
                                line_tokens = rag._count_tokens(line + '\n')
                                if temp_tokens + line_tokens <= TOKEN_BUDGET_EXTERNAL_KG_MAX:
                                    truncated_lines.append(line)
                                    temp_tokens += line_tokens
                                else:
                                    break
                            external_kg_context = '\n'.join(truncated_lines)
                        else:
                            external_kg_context = raw_external_kg

                        external_kg_tokens = rag._count_tokens(external_kg_context)

                        if external_kg_tokens > 0:
                            print(f"  ✅ [External KG] Retrieved {external_kg_tokens} tokens from external graph")
                            yield json.dumps({"type": "progress",
                                              "data": f"✅ External knowledge graph retrieval completed:{external_kg_tokens}tokens (P1 priority)"},
                                             ensure_ascii=False) + "\n"
                        else:
                            print(f"  ⚠️ [External KG] No valid facts found after truncation")
                    else:
                        print(f"  ⚠️ [External KG] No matching entities found in external graph")
                else:
                    print(f"  ⚠️ [External KG] No genes extracted from query, skipping")

            except Exception as e:
                print(f"  ❌ [External KG] Error: {e}. Skipping external graph.")
                log_to_file(f"[External KG] chat_endpoint error: {e}", "ERROR")
                import traceback
                log_to_file(traceback.format_exc(), "ERROR")
                external_kg_context = ""
                external_kg_tokens = 0
        # ========== End External KG ==========

        # ========== Graph RAG Enhancement (P2 Priority - Medium Confidence) ==========
        graph_context_str = ""
        graph_tokens = 0

        if USE_GRAPH_RAG and rag.graph_manager and final_res:
            try:
                yield json.dumps({"type": "progress", "data": "\U0001f578\ufe0f [GraphRAG] Build a knowledge graph and extract entity relationships..."},
                                 ensure_ascii=False) + "\n"
                print(f"  🕸️ [GraphRAG] Building knowledge graph from {len(final_res)} chunks...")

                # Note: `ents` has already been defined in the previous Entity Analysis step (no need to analyze again).
                # Build the graph and query it (reuse `ents` as seed nodes).
                raw_graph_context = await rag.graph_manager.build_and_query_graph(
                    final_res[:MAX_TRIPLET_CHUNKS],  # Use only the first N chunks
                    ents  # Reuse previously parsed entities as seeds for graph traversal
                )

                if raw_graph_context:
                    # Calculate the token count of the graph context
                    raw_graph_tokens = rag._count_tokens(raw_graph_context)

                    # If it exceeds the budget, truncate the graph context.
                    if raw_graph_tokens > MAX_GRAPH_CONTEXT_TOKENS:
                        print(f"  ✂️ [GraphRAG] Truncating: {raw_graph_tokens} -> {MAX_GRAPH_CONTEXT_TOKENS} tokens")
                        graph_context_str = rag._truncate_graph_context(raw_graph_context, MAX_GRAPH_CONTEXT_TOKENS)
                    else:
                        graph_context_str = raw_graph_context

                    graph_tokens = rag._count_tokens(graph_context_str) if graph_context_str else 0

                    if graph_tokens > 0:
                        print(f"  ✅ [GraphRAG] Generated graph context: {graph_tokens} tokens")
                        yield json.dumps({"type": "progress", "data": f"✅ Graph construction completed:{graph_tokens} tokens"},
                                         ensure_ascii=False) + "\n"
                    else:
                        print(f"  ⚠️ [GraphRAG] No valid graph context generated")
                else:
                    print(f"  ⚠️ [GraphRAG] No relationships extracted from documents")

            except Exception as e:
                print(f"  ❌ [GraphRAG] Error: {e}. Skipping graph context.")
                log_to_file(f"[GraphRAG] chat_endpoint error: {e}", "ERROR")
                graph_context_str = ""
                graph_tokens = 0
        # ========== End Graph RAG ==========

        # ========== Waterfall Token Budget Allocation ==========
        # Priority: P0 (external knowledge graph) > P1 (knowledge base) > P2 (graph) > P3 (literature)
        print(
            f"  ✅ [Diversity] Selected {len(final_res)} chunks (Threshold >= {SCORE_THRESHOLD})")

        # Step 1: Separate the synthesized knowledge base documents from the literature chunks
        kb_docs = [doc for doc in final_res if doc['metadata'].get('is_synthetic', False)]
        literature_docs = [doc for doc in final_res if not doc['metadata'].get('is_synthetic', False)]

        print(f"  🌊 [Waterfall] Token budget allocation:")
        print(f"       Total Budget: {MAX_CONTEXT_TOKENS} tokens")
        print(f"       System Reserved: {TOKEN_BUDGET_SYSTEM} tokens")
        print(f"       P0 - External KG: {external_kg_tokens} tokens (max {TOKEN_BUDGET_EXTERNAL_KG_MAX} tokens)")
        print(f"       P1 - KB Documents: {len(kb_docs)} (max {TOKEN_BUDGET_KB_MAX} tokens)")
        print(f"       P2 - Graph Context: {graph_tokens} tokens (max {TOKEN_BUDGET_GRAPH_MAX} tokens)")
        print(f"       P3 - Literature: {len(literature_docs)} chunks")

        papers_json = []
        context_parts = []
        current_token_count = TOKEN_BUDGET_SYSTEM  # Reserved system token

        # Header with date
        date_header = f"**Current Date:** {rag.current_date} (Use this to judge 'recent' evidence)\n"

        # ===== P1: Knowledge Base (Structured Data - Highest Priority) =====
        kb_context = ""
        kb_tokens = 0
        if kb_docs:
            kb_chunks = []
            kb_section_header = "\n=== [PRIORITY 1] DepMap Knowledge Base (Known SL Pairs) ===\n"
            kb_chunks.append(kb_section_header)

            for i, doc in enumerate(kb_docs):
                meta = doc['metadata']
                score_info = f"[Confidence: {doc['score']:.3f}]"
                chunk_text = (
                    f"\n[KB Entry {i + 1}] {score_info}\n"
                    f"{doc['content']}\n"
                )
                kb_chunks.append(chunk_text)

            kb_context = "".join(kb_chunks)
            kb_tokens = rag._count_tokens(kb_context)

            # If the KB exceeds the budget, truncate by confidence.
            if kb_tokens > TOKEN_BUDGET_KB_MAX:
                print(f"  ✂️ [P1-KB] Truncating: {kb_tokens} -> {TOKEN_BUDGET_KB_MAX} tokens")
                # Simple truncation strategy: keep documents from highest to lowest confidence until the budget is reached
                truncated_kb = [kb_section_header]
                temp_tokens = rag._count_tokens(kb_section_header)

                for i, doc in enumerate(sorted(kb_docs, key=lambda x: x['score'], reverse=True)):
                    chunk_text = f"\n[KB Entry {i + 1}] [Confidence: {doc['score']:.3f}]\n{doc['content']}\n"
                    chunk_tokens = rag._count_tokens(chunk_text)
                    if temp_tokens + chunk_tokens <= TOKEN_BUDGET_KB_MAX:
                        truncated_kb.append(chunk_text)
                        temp_tokens += chunk_tokens
                    else:
                        break

                kb_context = "".join(truncated_kb)
                kb_tokens = rag._count_tokens(kb_context)

            current_token_count += kb_tokens
            print(f"  ✅ [P1-KB] Allocated {kb_tokens} tokens for {len(kb_docs)} KB entries")

        # ===== P2: Graph RAG (Structured Relationships - Medium Priority) =====
        if graph_context_str and graph_tokens > 0:
            # The graph has already been truncated earlier to MAX_GRAPH_CONTEXT_TOKENS.
            # Check whether the waterfall budget still has remaining capacity.
            if current_token_count + graph_tokens > MAX_CONTEXT_TOKENS - 500:  # Reserve 500 tokens for references.
                print(f"  ⚠️ [P2-Graph] Budget tight, further truncating graph context")
                remaining_budget = max(500, MAX_CONTEXT_TOKENS - current_token_count - 500)
                if remaining_budget < graph_tokens:
                    graph_context_str = rag._truncate_graph_context(graph_context_str, remaining_budget)
                    graph_tokens = rag._count_tokens(graph_context_str)

            current_token_count += graph_tokens
            print(f"  ✅ [P2-Graph] Allocated {graph_tokens} tokens for knowledge graph")

        # ===== P0: External Knowledge Graph (Highest Priority - Structured Data) =====
        if external_kg_context and external_kg_tokens > 0:
            current_token_count += external_kg_tokens
            print(f"  ✅ [P0-External KG] Allocated {external_kg_tokens} tokens for external graph")

        # ===== P3: Literature (Unstructured Text - Lowest Priority) =====
        literature_token_budget = MAX_CONTEXT_TOKENS - current_token_count
        print(f"  ✂️ [P3-Literature] Remaining budget: {literature_token_budget} tokens")

        # Assemble the final headers (sorted by priority)
        final_header = date_header

        # P0: External knowledge graph (highest priority)
        if external_kg_context:
            final_header += "\n" + external_kg_context + "\n"

        # P1: DepMap KB
        if kb_context:
            final_header += kb_context

        # P2: Graph RAG
        if graph_context_str:
            final_header += "\n" + graph_context_str + "\n"

        # P3: Literature header
        final_header += "\n=== [PRIORITY 3] Retrieved Scientific Literature ===\n"

        context_parts.append(final_header)
        current_token_count = rag._count_tokens(final_header)

        for i, res in enumerate(literature_docs):
            meta = res['metadata']
            title = meta.get('paper_title', 'Unknown')
            score_info = f"[Rel: {res['score']:.3f}]"

            if meta.get('is_web'):
                source_type = f"🌐 [WEB] ({meta.get('link')})"
                genes_info = ""
            else:
                source_type = "📄 [LOCAL]"
                genes = ", ".join(meta.get('key_genes', [])[:3])
                genes_info = f"Key Genes: {genes}\n" if genes else ""

            chunk_text = (
                f"\nSource [{i + 1}] {source_type} {score_info}\n"
                f"Title: {title}\n{genes_info}Content: {res['content']}\n"
            )

            tokens = rag._count_tokens(chunk_text)
            if current_token_count + tokens > literature_token_budget:
                print(f"      🛑 [Context] Limit reached at Source [{i + 1}].")
                break

            context_parts.append(chunk_text)
            current_token_count += tokens

            # Decide whether to truncate content based on EVAL_MODE.
            if EVAL_MODE:
                # Evaluation mode: return the full content (or limited to EVAL_CONTENT_MAX_LENGTH)
                content_for_json = res['content']
                if EVAL_CONTENT_MAX_LENGTH > 0 and len(content_for_json) > EVAL_CONTENT_MAX_LENGTH:
                    content_for_json = content_for_json[:EVAL_CONTENT_MAX_LENGTH] + "..."
            else:
                # Production mode: truncate to 200 characters to reduce response size.
                content_for_json = res['content'][:200] + "..."

            papers_json.append({
                "id": i + 1,
                "title": title,
                "score": res['score'],
                "content": content_for_json,
                "link": meta.get('link', '')
            })

        context_str = "".join(context_parts)
        print(
            f"    ✅ [Context] Final composition:")
        print(f"         - P0 External KG: {external_kg_tokens} tokens")
        print(f"         - P1 KB: {kb_tokens} tokens ({len(kb_docs)} entries)")
        print(f"         - P2 Graph: {graph_tokens} tokens")
        print(f"         - P3 Literature: {len(papers_json)} chunks")
        print(f"         - Total: ~{current_token_count} tokens (Budget: {MAX_CONTEXT_TOKENS})")

        # Check Relevance Threshold
        if not final_res or final_res[0]['score'] < SCORE_THRESHOLD:
            yield json.dumps({"type": "references", "data": []}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "progress", "data": "\u26a0\ufe0f Search results have low relevance; switching to general knowledge Q&A..."},
                             ensure_ascii=False) + "\n"
            # Fallback prompt
            prompt = f"User Query: {user_query}\nNo specific papers found in database. Answer based on general scientific knowledge."
            context_str = "No Context."
        else:
            yield json.dumps({"type": "references", "data": papers_json}, ensure_ascii=False) + "\n"
            prompt = f"{SOLVER_PROMPT}\n\nQuestion: {user_query}"

        # 4. Solver Generation (Draft)
        yield json.dumps({"type": "progress", "data": "\U0001f9ea Solver is building a preliminary response..."}, ensure_ascii=False) + "\n"

        log_to_file("=" * 80)
        log_to_file(f"[Solver] Start generating the answer")
        log_to_file(
            f"[Solver] Context length:{len(context_str)}Characters, token count:{current_token_count}")
        log_to_file(
            f"[Solver] Graph Context: {graph_tokens} tokens included" if graph_tokens > 0 else "[Solver] Graph Context: None")
        log_to_file(f"[Solver] Prompt: {prompt[:200]}...")

        messages = [
            {"role": "system", "content": "You are a scientific assistant."},
            {"role": "user", "content": f"Context:\n{context_str}\n\n{prompt}"}
        ]

        log_to_file(f"[Solver] Full User message length:{len(messages[1]['content'])}Character")

        candidate_answer = ""
        response = await client.chat.completions.create(
            model=SOLVER_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            stream=True
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                candidate_answer += chunk.choices[0].delta.content

        log_to_file(f"[Solver] Generated answer length:{len(candidate_answer)}Character")
        log_to_file(f"[Solver] Answer Summary:{candidate_answer[:300]}...")
        log_to_file("=" * 80 + "\n")

        # 5. Verification Loop
        max_retries = 5
        final_answer = candidate_answer

        for i in range(max_retries):
            yield json.dumps({"type": "progress", "data": f"🔍 Verifier is performing the{i + 1}Round of review..."},
                             ensure_ascii=False) + "\n"

            log_to_file("=" * 80)
            log_to_file(f"[Verifier] No.{i + 1}Round verification started")

            verify_input = f"Context:\n{context_str}\n\nCandidate:\n{candidate_answer}\n\n{VERIFIER_PROMPT}"
            log_to_file(f"[Verifier] Validate input length:{len(verify_input)}Character")
            log_to_file(f"[Verifier] Candidate answer:{candidate_answer[:200]}...")

            v_resp = await client.chat.completions.create(
                model=VERIFIER_MODEL_NAME,
                messages=[{"role": "user", "content": verify_input}],
                temperature=0.4
            )
            critique = v_resp.choices[0].message.content

            log_to_file(f"[Verifier] Verification result:{critique}")

            yield json.dumps({"type": "thinking", "data": f"[Critique Round {i + 1}]\n{critique}\n"},
                             ensure_ascii=False) + "\n"

            # Use more lenient validation logic (consistent with sl_agent_online2.py).
            critique_upper = critique.upper()
            is_pass = False
            if "VERDICT: PASS" in critique_upper or "VERDICT:PASS" in critique_upper:
                is_pass = True
            elif "VERDICT: FAIL" in critique_upper or "VERDICT:FAIL" in critique_upper:
                is_pass = False
            else:
                # Backup plan: detect whether the model has mistakenly entered analysis mode
                first_50 = critique[:50].lower()
                if any(word in first_50 for word in ["okay", "let's", "the user", "looking at", "###"]):
                    is_pass = False
                else:
                    is_pass = True

            if is_pass:
                log_to_file(f"[Verifier] Verdict: PASS")
                log_to_file("=" * 80 + "\n")
                yield json.dumps({"type": "progress", "data": "\u2705 Verification passed! Preparing output..."}, ensure_ascii=False) + "\n"
                final_answer = candidate_answer
                break
            else:
                log_to_file(f"[Verifier] Decision: FAIL, starting correction")
                yield json.dumps({"type": "progress", "data": "\u26a0\ufe0f Logic flaw detected, self-correcting..."},
                                 ensure_ascii=False) + "\n"

                refine_input = f"Previous Answer:\n{candidate_answer}\n\nCritique:\n{critique}\n\nContext:\n{context_str}\n\nFix the answer strictly."
                log_to_file(
                    f"[Solver-Refine] Starting correction, input length:{len(refine_input)}Character")

                r_resp = await client.chat.completions.create(
                    model=SOLVER_MODEL_NAME,
                    messages=[{"role": "user", "content": refine_input}],
                    temperature=0.3
                )
                candidate_answer = r_resp.choices[0].message.content
                final_answer = candidate_answer

                log_to_file(
                    f"[Solver-Refine] Corrected answer length:{len(candidate_answer)}Character")
                log_to_file(
                    f"[Solver-Refine] Corrected Answer Summary:{candidate_answer[:300]}...")
                log_to_file("=" * 80 + "\n")

        # 6. Stream Final Answer
        yield json.dumps({"type": "progress", "data": "\U0001f4dd Generating final report..."}, ensure_ascii=False) + "\n"

        chunk_size = 15  # Slightly increase the block size to make the output smoother.
        for i in range(0, len(final_answer), chunk_size):
            yield json.dumps({"type": "token", "data": final_answer[i:i + chunk_size]}, ensure_ascii=False) + "\n"
            import asyncio
            await asyncio.sleep(0.005)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    # Port: 16006
    uvicorn.run(app, host="0.0.0.0", port=16006)
