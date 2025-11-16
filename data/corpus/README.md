This directory contains **example datasets** that demonstrate the input format and output structure of the Agent-Trace Corpus (ATC).

<div align="center" style="margin: 0px 0 30px; padding: 0px 10px 10px; color: #856404; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: .25rem;">

### 🚧 Important Notice 🚧

These datasets are **excerpts** from the full-sized datasets used in our research. They are provided here for: (1) testing the simulation pipeline, (2) understanding the expected input/output format, and (3) quick experimentation and validation

</div>

## Full Dataset Access

The **complete datasets** with is being made available on Zenodo.  *Coming soon*

# Agent-Trace Corpus (ATC-47k) 

**Version**: 0.1.0  
**Generated**: November 2025  
**Platform**: AgentSim

A unified, training-ready corpus of agent reasoning traces from 3,000 exploratory simulations across three datasets (MSMARCO, Quasar-T, CausalQA) using three SOTA models (GPT-4o, Mistral-Large, DeepSeek-V3).

---

## Corpus Statistics (IN-PROGRESS)

| Metric | Count |
|--------|-------|
| **Reasoning Trace Steps** | 46,840 |
| **Trajectory Steps** | 46,837 |
| **Supervised Training Pairs** | 10,221 |
| **Unique Documents** | 100,342 |
| **Total Queries** | 12,220 |
| **Datasets** | MSMARCO, Quasar-T, CausalQA |
| **Models** | GPT-4o, Mistral-Large, DeepSeek-V3 |

---

## Corpus Structure

```
corpus/
├── README.md                        # This file
├── corpus_stats.json               # Detailed statistics
│
├── traces/
│   └── all_traces.jsonl.gz        # Complete reasoning traces (46.8K steps)
│
├── trajectories/
│   └── all_trajectories.jsonl.gz  # High-level trajectories (46.8K steps)
│
├── supervised/
│   └── all_supervised.jsonl.gz    # Query-document-answer pairs (10.2K)
│
├── retrievals/
│   └── all_retrievals.json.gz     # Document retrieval logs (100.3K docs)
│
└── queries/
    └── all_queries.json.gz        # All queries executed (12.2K)
```

---

## Training Use Cases

### 1. **Chain-of-Thought Training**
Use `traces/all_traces.jsonl.gz` for training models on step-by-step reasoning:
- Each trace contains: thought → action → observation
- Includes LLM input/output for each step
- Teacher and consultant model outputs for comparison

### 2. **Imitation Learning**
Use `supervised/all_supervised.jsonl.gz` for supervised fine-tuning:
- Query-document-answer triples
- Reasoning steps showing how answer was derived
- Multi-hop reasoning examples

### 3. **Retrieval-Augmented Generation**
Use `trajectories/all_trajectories.jsonl.gz` for RAG training:
- High-level decision sequences
- Document retrieval patterns
- Query reformulation strategies

### 4. **Query Reformulation**
Use `queries/all_queries.json.gz` for query expansion models:
- Original queries and reformulations
- Semantic vs syntactic reformulation patterns
- Model-specific strategies

### 5. **Retrieval System Evaluation**
Use `retrievals/all_retrievals.json.gz` for IR research:
- Document frequency across runs
- Coverage analysis
- Redundancy patterns

---

## Data Format

### Reasoning Traces (`traces/all_traces.jsonl.gz`)

Each line is a JSON object representing one execution step:

```json
{
  "turn": 1,
  "component": "retriever",
  "thought": "I need to find information about...",
  "action": {"tool": "chatnoir", "parameters": {"query": "...", "top_k": 20}},
  "observation": {"documents_retrieved": 20},
  "llm_input": "Given the query...",
  "llm_output": "I should retrieve...",
  "tool_output": {
    "teacher_output": "...",
    "consultant_outputs": [...]
  },
  "execution_time_ms": 234.5,
  "tokens": 450,
  "_source": {
    "run_id": "6601a11b",
    "dataset": "msmarco",
    "sample_id": "sample_001"
  }
}
```

### Supervised Pairs (`supervised/all_supervised.jsonl.gz`)

Training pairs with reasoning chains:

```json
{
  "query": "what was the immediate impact of the manhattan project?",
  "documents": [
    {"id": "doc_123", "text": "...", "rank": 1},
    {"id": "doc_456", "text": "...", "rank": 2}
  ],
  "answer": "The immediate impact was...",
  "reasoning_steps": [
    "Retrieved 20 documents",
    "Identified key impacts",
    "Synthesized answer"
  ],
  "metadata": {
    "model": "gpt-4o",
    "dataset": "msmarco",
    "timestamp": "2025-11-12T10:30:00Z"
  },
  "_source": {...}
}
```

### Trajectories (`trajectories/all_trajectories.jsonl.gz`)

High-level action sequences:

```json
{
  "turn": 1,
  "component": "retriever",
  "action": "retrieve_documents",
  "parameters": {"query": "manhattan project impact"},
  "result": "success",
  "documents_retrieved": 20,
  "execution_time_ms": 234.5,
  "_source": {...}
}
```

---

## Quick Start

### Load Reasoning Traces

```python
import gzip
import json

traces = []
with gzip.open('traces/all_traces.jsonl.gz', 'rt') as f:
    for line in f:
        traces.append(json.loads(line))

print(f"Loaded {len(traces):,} trace steps")
```

### Load Supervised Pairs

```python
import gzip
import json

pairs = []
with gzip.open('supervised/all_supervised.jsonl.gz', 'rt') as f:
    for line in f:
        pairs.append(json.loads(line))

print(f"Loaded {len(pairs):,} training pairs")
```

### Filter by Dataset

```python
# Get only CausalQA traces
causalqa_traces = [
    t for t in traces 
    if t['_source']['dataset'] == 'causalqa'
]
```

### Filter by Model

```python
# Get only GPT-4o traces (run IDs: 6601a11b, 14b8e660, 4dc6e14f)
gpt4o_runs = ['6601a11b', '14b8e660', '4dc6e14f']
gpt4o_traces = [
    t for t in traces 
    if t['_source']['run_id'] in gpt4o_runs
]
```

---

## Corpus Metadata

All entries include `_source` metadata for traceability:

```json
{
  "_source": {
    "run_id": "6601a11b",        // Unique run identifier
    "dataset": "msmarco",         // Dataset: msmarco, quasart, causalqa
    "sample_id": "sample_001"     // Sample identifier (traces/trajectories only)
  }
}
```

### Run ID → Model Mapping

| Dataset | Run ID | Teacher Model |
|---------|--------|---------------|
| MSMARCO | 6601a11b | GPT-4o |
| MSMARCO | b1b576a8 | DeepSeek-V3 |
| MSMARCO | fb969481 | Mistral-Large |
| Quasar-T | 14b8e660 | GPT-4o |
| Quasar-T | 7674b991 | DeepSeek-V3 |
| Quasar-T | d5e9bf86 | Mistral-Large |
| CausalQA | 4dc6e14f | GPT-4o |
| CausalQA | 4c8f12ff | DeepSeek-V3 |
| CausalQA | bb0fafe9 | Mistral-Large |
| CausalQA | f4664f84 | GPT-4o | 

