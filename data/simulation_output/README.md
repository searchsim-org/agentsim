# Simulation Traces

Simulation runs write structured artifacts to this directory. Each template has its own subdirectory, and every execution receives a unique run identifier (timestamp or UUID) to ensure outputs are reproducible and comparable.

---

## Directory Layout
- `<template_id>/` – named after the simulation template (for example `standard_gpt4o` or `exploratory_gpt4o`).
- `<template_id>/<run_id>/` – a single execution; contains summary JSON plus per-sample folders when applicable.
- `<template_id>/<run_id>/<dataset>/sample_<index>/` – granular exports for each query or seed processed during the run.

The run directory mirrors the structure produced by `TraceExporter` (`agentsim/utils/trace_exporter.py`).

---

## Core Files
- `manifest.json` – entry point with metadata: run UUID, teacher model, template name, total tokens, estimated cost, and pointers to other files.
- `config.json` – fully resolved simulation configuration combining template defaults, environment overrides, and runtime adjustments.
- `stats.json` – aggregated metrics (iterations, success flags, error counts, latency, token usage).
- `coverage.json` – coverage curves showing cumulative unique documents and overlap across seeds.
- `documents_retrieved.json` – retrieval hits with scores per query; useful for diagnosing backend performance.
- `exploration_log.json` – step-by-step record of decisions in exploratory mode (questions asked, reflections, component choices).
- `knowledge_base.json` – persistent facts assembled during exploration.
- `queries_explored.json` – original seeds plus follow-up questions synthesized on the fly.
- `traces.jsonl` – canonical step-level trace designed for fine-tuning or offline evaluation pipelines.
- `trajectories.jsonl` – episode-oriented view synthesizing each query/run.
- `supervised.jsonl` – optional teacher responses formatted for supervised training.
- `manifest.yaml` or additional audit files – present when the template or exporter produces extra summaries.

Per-sample directories repeat some of these files (such as `traces.jsonl`) scoped to a single query.

---

## Inspecting Outputs
1. Open the most recent run directory:
   ```bash
   ls data/simulation_output/<template>/
   ```
2. Use `manifest.json` to locate files of interest and fetch top-level metrics.
3. Load `traces.jsonl` or per-sample JSONL files with tools like `jq`, pandas, or the `agentsim.utils.trace_exporter.TraceExporter` helper class.
4. Compare runs by aligning on run IDs and diffing `coverage.json` or `stats.json` metrics.

Example (extract average latency):
```bash
jq '.total_latency_ms' data/simulation_output/standard_gpt4o/<run_id>/stats.json
```

---

## Downstream Analysis
- **Training Data Preparation:** Feed `traces.jsonl` and `supervised.jsonl` into downstream pipelines for SFT or reinforcement learning by streaming each line into your preferred format converter.
- **Coverage Reporting:** Plot `coverage.json` values over iterations to evaluate how well retrieval expands the document set.
- **Prompt Auditing:** Review `exploration_log.json` to understand teacher reflections, follow-up prompts, and decision-making.
- **Comparative Studies:** Archive `manifest.json` files from multiple runs and compare token usage, success counts, and average similarity scores.

For automated processing, instantiate `TraceExporter` with a custom output directory and call `export_run` or `export_summary_files` from inside your simulation hooks.

---

## Cleanup and Retention
- Delete individual runs after analysis with:
  ```bash
  rm -rf data/simulation_output/<template>/<run_id>
  ```
- Preserve long-term experiments by compressing the run directory or exporting summaries to object storage.
- Add `.gitignore` rules (already present) to prevent large trace directories from being committed while keeping this README and `.gitkeep` in version control.

---

Refer back to the root `README.md` for guidance on running new simulations and interpreting these artifacts in the broader workflow.

