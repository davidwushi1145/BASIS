# BASIS: An LLM-Empowered Benchmark and Agent System for Inferring Synthetic Lethality

This repository is anonymized for double-blind review.

**Keywords**: Synthetic Lethality, Neuro-symbolic AI, Benchmarking, LLM Agents, AI4Science

## Abstract

Synthetic lethality (SL) has emerged as a clinically validated approach to identifying anticancer therapeutic targets, effectively exposing the critical vulnerabilities unique to cancer cells. While Large Language Models (LLMs) have achieved significant progress in scientific research, their reliability in reasoning about SL gene pairs remains unexplored.  To fill this gap, we present a unified framework, named BASIS, for evaluating and improving LLMs in inferring SL interactions for anticancer drug target discovery. We first construct SL-Bench, a systematic benchmark encompassing seven hierarchical tasks, to rigorously evaluate LLMs across a spectrum of cognitive demands ranging from factual recall to clinical hypothesis formulation, thereby probing their capacity for deep mechanistic reasoning over shallow statistical correlations. Our benchmarking study revealed a critical limitation: contemporary LLMs suffer from substantial logical inconsistencies when inferring intricate biological dependencies. To resolve this issue, we propose SL-Agent, a neuro-symbolic Solver-Verifier architecture that integrates GraphRAG with iterative self-reflection, demonstrating consistent performance improvements when applied to various base models. Notably, on the most challenging task, an 8B open-source model equipped with SL-Agent outperforms leading proprietary models, exceeding GPT-5.2 in accuracy by 14.0 pp and Gemini-3-Pro in safety by 28.0 pp. By unifying systematic evaluation with targeted improvement, BASIS establishes a practical path toward reliable, AI-driven synthetic lethality discovery.

## Repository Contents (Two Components)

- **SL-Bench (benchmark)**: `sl_bench/task/` contains JSONL task files.
- **SL-Agent (agent system)**: implementation of the retrieval, reasoning, and verification pipeline (`api/`, `core/`, `retrieval/`, `tools/`, `llm/`, `models/`, `config/`).

## SL-Bench: Tasks and Data

Each example is one JSON object per line (`.jsonl`) with:

- `input`: the prompt presented to the model
- `target`: the gold label (`Yes`/`No` or a single option letter)
- `metadata`: task-specific structured fields

### Task Files (As Released in This Repository)

| Task | File | Output space |    N |
| --- | --- | --- |-----:|
| A1: Binary SL identification | `sl_bench/task/task_a1_classification.jsonl` | `Yes`/`No` | 1000 |
| A2: Conditional partner retrieval | `sl_bench/task/task_a2_partner_retrieval.jsonl` | `A`/`B`/`C`/`D` |  300 |
| B1: SL Mechanism Deduction | `sl_bench/task/task_b1_sl_binary_classification.jsonl` | `Yes`/`No` |  200 |
| B2: SL vs. SDL distinction | `sl_bench/task/task_b2_sl_sdl_distinction.jsonl` | `A`/`B`/`C`/`D` | 400 |
| C1: Counterfactual consistency | `sl_bench/task/task_c1_counterfactual.jsonl` | `A`/`B`/`C`/`D` |  200 |
| C2: Alias robustness | `sl_bench/task/task_c2_alias_robustness.jsonl` | `Yes`/`No` |  300 |
| D1: Hypothesis Formulation (exploratory) | `sl_bench/task/task_d1_clinical_recommendation.jsonl` | `A`/`B`/`C`/`D` |  200 |

### Task Definitions (Short)

- **A1**: given a gene pair, predict whether it is synthetic lethal.
- **A2**: given a loss-of-function alteration context, select the correct SL partner from multiple choices.
- **B1**: given single/double perturbation readouts, classify SL based on synergy rather than phenotype magnitude.
- **B2**: distinguish classic SL (double loss) from synthetic dosage lethality (overexpression plus inhibition).
- **C1**: test whether the model updates its answer when a counterfactual premise contradicts known biology.
- **C2**: test whether answers are invariant under gene alias/obsolete-name substitution.
- **D1**: given a de-identified patient genomic profile, select the most plausible SL-based therapeutic hypothesis (research-only).

### Construction Overview (High-Level)

As described in the paper, SL-Bench is designed to stress scientific reasoning rather than surface correlation. Key design elements include:

- Multi-source integration of validated SL evidence, functional dependency signals, and gene/drug annotations
- Difficulty-aware negatives (random, co-expression hard negatives, and pathway-based negatives)
- Robustness stress tests via counterfactual premise injection (C1) and gene alias perturbations (C2)
- A translational, hypothesis-formulation task (D1) intended for research benchmarking only

### Minimal Dataset Example

```json
{"input":"Is there a synthetic lethal relationship between ACADSB and ACADL?","target":"No","metadata":{"gene_a":"ACADSB","gene_b":"ACADL","label":0,"negation_type":"Level3_SamePathway"}}
```

## SL-Agent: System Overview

SL-Agent is a neuro-symbolic pipeline designed to improve scientific reliability under evidence constraints. The key modules implemented in this codebase include:

- Intent-aware routing and query planning
- Gene synonym expansion and biomedical entity alignment
- Hybrid retrieval (dense + sparse) with optional multi-hop "step-back" retrieval when initial evidence is weak
- Priority-based context assembly (evidence hierarchy)
- Solver-Verifier loop for grounded drafting and iterative self-correction

![sl-agent](https://raw.githubusercontent.com/davidwushi1145/photo2/main/202603022149959.jpg)

## Data Availability

The benchmark task files (`sl_bench/task/*.jsonl`) are included in this repository. Running SL-Agent and reproducing the full retrieval stack additionally requires several data artifacts (listed below). Download links is https://drive.google.com/file/d/16U_F3N3fL8QL5ALR9rhyaYkYGha6NVJB/view?usp=sharing.

### Data Artifacts

Recommended layout (example):

```text
data/
  rag_corpus_contextual.jsonl
  rag_corpus_contextual_vectors.pt
  kg.db
  SL_Benchmark_Final.csv
  CRISPRGeneDependency.csv
  sl_master_dataset.csv
```

### Environment Variables (Minimal Example)

```bash
CORPUS_FILE=./data/rag_corpus_contextual.jsonl
VECTOR_CACHE=./data/rag_corpus_contextual_vectors.pt
DEPMAP_DATA_DIR=./data
EXTERNAL_KG_DB_PATH=./data/kg.db
```

### Notes on Third-Party Data

Some artifacts are derived from third-party resources (e.g., functional dependency data and curated SL evidence) and may be subject to their original terms. Users should ensure compliance with upstream licenses when re-distributing raw sources.

## Running SL-Agent (API Server)

### Installation

```bash
poetry install
```

### Configuration

Copy `.env.example` to `.env` and fill in required paths and API keys.

```bash
cp .env.example .env
```

### Start the Server

```bash
make run
```

### Query the API

`POST /chat` streams NDJSON events.

```bash
curl -sN \
  -H "Content-Type: application/json" \
  -d '{"query":"What are synthetic lethal partners of BRCA1?"}' \
  http://127.0.0.1:6006/chat
```

## Benchmarking Helpers

This repository provides SL-Bench task files under `sl_bench/task/`. If you evaluate models on these tasks, ensure that the model output is constrained to the required answer space (`Yes`/`No` or a single option letter) before computing exact-match accuracy.

**Metric (default)**: exact-match accuracy against `target` after output normalization.

## Safety and Ethics Notes

Task D1 is hypothesis-generating and for research benchmarking only. Outputs must not be interpreted as clinical recommendations; wet-lab and clinical validation are required.

## License

See `LICENSE.txt`.
