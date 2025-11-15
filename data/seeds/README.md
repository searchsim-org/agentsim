# Seed Files

Seed files provide curated query starting points for exploratory and adaptive simulations. Each line in a seed file is a JSON object representing the initial query (and optional metadata) that the simulator will expand on.

---

## Directory Layout
- `seeds_<dataset>_<split>_<size>.jsonl` – curated batches aligned with specific datasets.
- `*_metadata.json` (optional) – statistics produced by the seed selector (cluster coverage, novelty thresholds, etc.).
- Custom files you generate should follow the same naming convention to make provenance obvious.

---

## Seed Record Format
Minimal entry:
```json
{"query": "How does retrieval augmented generation work?"}
```
Optional fields supported by templates and components:
- `context` – supplemental information to inject into the initial turn.
- `answers` – authoritative references used during evaluation.
- `metadata` – arbitrary fields used in custom components.

---

## Using Seeds in a Simulation Template
```yaml
datasets:
  - name: msmarco_seeds
    path: ./data/seeds/seeds_msmarco_train_1k.jsonl
    num_samples: 1000          # optional subsampling
    sample_strategy: sequential
```
Tips:
- When `num_samples` is omitted, the entire file is consumed.
- Use `sample_strategy: random` to shuffle the file each run.
- Multiple seed files can be listed if you want to compare corpora.

Launch the template:
```bash
poetry run agentsim simulate exploratory_seeds_opensearch
```
Outputs land in `data/simulation_output/<template>/<run_id>/`.

---

## Generating New Seed Batches
The `agentsim seed-select` command produces coverage-aware seeds from datasets in `data/datasets/`.  
Pick the recipe that matches your corpus:

### MSMARCO + OpenSearch (default)
```bash
poetry run agentsim seed-select \
  --data-dir ./data/datasets \
  --dataset msmarco \
  --split train \
  --retrieval opensearch \
  --index msmarco-v2.1-segmented \
  --topk 20 \
  --max-candidates 50000 \
  --num-seeds 1000 \
  --clusters 100 \
  --novelty-threshold 0.6 \
  --lambda-mmr 0.7 \
  --output ./data/seeds/seeds_msmarco_train_1k.jsonl
```
Environment:
- `OPENSEARCH_HOST` (`https://...`) and (optionally) `OPENSEARCH_PORT`
- `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD` if the cluster requires auth
- `OPENSEARCH_USE_SSL=true` for HTTPS verification

### Quasar-T (or similar) + ChatNoir
```bash
poetry run agentsim seed-select \
  --data-dir ./data/datasets \
  --dataset quasar-t \
  --split train \
  --retrieval chatnoir \
  --index cw12 \
  --topk 10 \
  --max-candidates 2000 \
  --num-seeds 200 \
  --clusters 50 \
  --novelty-threshold 0.6 \
  --lambda-mmr 0.7 \
  --request-delay 0.3 \
  --output ./data/seeds/seeds_quasart_train.jsonl
```
Environment:
- `CHATNOIR_API_KEY` (required)
- `CHATNOIR_BASE_URL` (optional, defaults to `https://www.chatnoir.eu/api/v1`)

### CausalQA + ChatNoir (Large Dataset)
```bash
poetry run agentsim seed-select \
  --data-dir ./data/datasets \
  --dataset causalqa \
  --split train \
  --retrieval chatnoir \
  --index cw12 \
  --topk 20 \
  --max-candidates 15000 \
  --num-seeds 1000 \
  --clusters 100 \
  --novelty-threshold 0.6 \
  --lambda-mmr 0.7 \
  --request-delay 0.3 \
  --output ./data/seeds/seeds_causalqa_train.jsonl
```
Notes:
- `--max-candidates 15000` limits queries loaded from the dataset 
- The clustering algorithm ensures diverse coverage even with a subset
- For broader coverage, run multiple batches with `--prior-docs-file` to avoid overlap

Tips:
- Use `--request-delay ≥ 0.25` with `chatnoir` to avoid rate limits.
- Reduce `--max-candidates` and `--num-seeds` for quick dry runs.
- Increase `--clusters` for broader topical coverage when you scale up.

Key parameters for both flows:
- `--clusters` – higher values increase topical diversity.
- `--novelty-threshold` – lower values enforce stricter overlap filtering.
- `--lambda-mmr` – balances relevance and diversity within `0.0–1.0`.
- `--prior-docs` – (optional) JSONL with doc IDs already covered in previous runs.

---

## Best Practices for Seed Design
- **Balance coverage and novelty:** Monitor `coverage.json` metrics after a simulation and adjust `--clusters` or `--novelty-threshold` accordingly.
- **Align with evaluation goals:** For teacher comparisons, reuse the same seed batch across models to obtain fair metrics.
- **Document provenance:** Store the command (or a short README note) alongside each batch to record how it was generated.
- **Validate format:** Quick check with `head` or `jq` before committing.
- **Version large batches externally:** For very large seed sets, consider storing them in object storage and referencing them via `--seeds` CLI argument to avoid bloating the repository.

---

## Updating Templates After Regeneration
1. Replace the JSONL file here or add a new file with a descriptive name.
2. Update any simulation template to point to the new path.
3. Run `poetry run agentsim validate <template>` to confirm the dataset block resolves correctly.
4. Commit the new seeds and templates if they are part of your shared setup.

---

Consult the root `README.md` for a broader overview of how seeds interact with datasets, retrieval backends, and simulation templates.

