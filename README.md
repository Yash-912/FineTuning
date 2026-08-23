# Prompt Injection Detector

An end-to-end ML system that detects prompt injection attacks in LLM traffic. It fine-tunes **Qwen2-1.5B with QLoRA** as a binary sequence classifier (benign vs. injection), calibrates its confidence with temperature scaling, and serves it as a **FastAPI reverse proxy** that screens every message before forwarding traffic to any OpenAI-compatible LLM endpoint. The system ships with Prometheus/Grafana monitoring, shadow-mode evaluation, a human review queue, and an automated retraining loop.

## How It Works

```
                 ┌──────────────────────────────────────────────────────┐
                 │                     TRAINING PIPELINE                │
                 │                                                      │
  HuggingFace ──►│  loader.py ──► deduplicator.py ──► balancer.py       │
  datasets (5)   │                                      │               │
                 │                          stratified_split (80/10/10) │
                 │                                            │         │
                 │              train_qlora.py ◄──────────────┘         │
                 │  Qwen2-1.5B + 4-bit NF4 + LoRA (SEQ_CLS)             │
                 │                        │                             │
                 │                   calibrate.py                       │
                 │            temperature scaling on val NLL            │
                 └────────────────────────┼─────────────────────────────┘
                                          ▼
                    models/qwen-injection-detector/best
                    (LoRA adapter + tokenizer + temperature.pt)
                                          │
                 ┌────────────────────────▼─────────────────────────────┐
                 │                  SERVING PIPELINE                    │
                 │                                                      │
  Client ───────►│  FastAPI :8080  /chat/completions                    │
  (OpenAI SDK)   │        │                                             │
                 │        ├── InjectionClassifier.predict()             │
                 │        │   (4-bit Qwen2 + LoRA + temperature)        │
                 │        │                                             │
                 │        ├── label=1 & conf ≥ threshold?               │
                 │        │     hard_block → 403 JSON error             │
                 │        │     soft_flag  → X-Injection-Detected hdr   │
                 │        │                                             │
                 │        ├── shadow classifier (optional, parallel)    │
                 │        └── ReviewQueue (SQLite WAL) logs everything  │
                 │                                                      │
                 │  /health   → readiness + mode + threshold            │
                 │  /metrics  → Prometheus exposition                   │
                 └────────────────────────┬─────────────────────────────┘
                                          ▼
                            Upstream LLM API (OpenAI-compatible)
```

## Repository Layout

| Path | Purpose |
|---|---|
| `configs/dataset_config.yaml` | Data source registry, dedup/balance/split parameters |
| `configs/training_config.yaml` | Base model, quantization, LoRA, and Trainer hyperparameters |
| `configs/prometheus.yml` | Prometheus scrape config (5s interval against middleware `:8080/metrics`) |
| `configs/grafana/` | Provisioned Grafana datasource and "Injection Detector" dashboard |
| `docker-compose.yml` | Prometheus (`:9090`) + Grafana (`:3000`) containers |
| `src/data/loader.py` | Per-source HuggingFace dataset loaders (normalize to `text`, `label`, `source`) |
| `src/data/deduplicator.py` | Exact dedup + embedding-based near-dedup (cosine similarity) |
| `src/data/balancer.py` | Benign:injection ratio enforcement and stratified train/val/test split |
| `src/utils/config.py` | YAML config loader |
| `scripts/prepare_dataset.py` | Full data pipeline: load → dedup → balance → split → parquet + dataset card |
| `scripts/train_qlora.py` | QLoRA fine-tuning of Qwen2-1.5B for sequence classification |
| `scripts/calibrate.py` | Temperature scaling via grid search + bounded NLL minimization |
| `scripts/evaluate_baseline.py` | Zero-shot evaluation of `deepset/deberta-v3-base-injection` on the test set |
| `scripts/evaluate_model.py` | Test-set evaluation of the fine-tuned model + baseline comparison + confusion matrix plot |
| `scripts/build_adversarial_eval.py` | 47 hand-authored adversarial examples across 12 attack categories |
| `scripts/eval_adversarial.py` | Replays the adversarial set through the live middleware HTTP API |
| `middleware/app.py` | FastAPI reverse proxy with hard-block/soft-flag modes and Prometheus metrics |
| `middleware/classifier.py` | Singleton `InjectionClassifier` (4-bit base + LoRA adapter + temperature) |
| `middleware/config.py` | Pydantic settings loaded from environment / `.env` |
| `middleware/example_client.py` | One-line OpenAI SDK integration demo |
| `middleware/test_middleware.py` | pytest integration tests (FastAPI `TestClient`) |
| `scripts/review_queue.py` | SQLite-backed human labeling queue + interactive CLI |
| `scripts/retrain.py` | Continuous retraining: merge labels → retrain → recalibrate → promote to shadow |
| `eval/` | Generated metrics JSON, comparison, calibration curve, confusion matrices |
| `data/processed/` | Train/val/test/adversarial parquet files + dataset card |

## Dataset Pipeline

**Sources** (registered in `configs/dataset_config.yaml`, all pulled from HuggingFace):

| Source | Role |
|---|---|
| `deepset/prompt-injections` | Mixed benign + injection |
| `S-Labs/prompt-injection-dataset` | Mixed benign + injection |
| `xTRam1/safe-guard-prompt-injection` | Injection-only (loader keeps `label == 1` rows) |
| `Lakera/gandalf_ignore_instructions` | Injection-only, subsampled to 350 |
| `HuggingFaceH4/no_robots` | Benign-only (first user turn of each conversation) |
| `hackaprompt/hackaprompt-dataset` | Injection-only, disabled in config; filters to successful submissions when enabled |

**Processing**:
1. Each source is normalized to `(text, label, source)` — `label=0` benign, `label=1` injection.
2. **Exact dedup**: lowercase/stripped text, duplicates resolved by source priority order.
3. **Near-dedup**: `all-MiniLM-L6-v2` sentence embeddings, cosine similarity ≥ **0.92** flagged; lower-priority source row dropped.
4. **Balancing**: benign downsampled to a **2:1 benign:injection** target ratio.
5. **Split**: stratified by source into **80% / 10% / 10%** train/val/test (seed 42).
6. Outputs `train.parquet`, `val.parquet`, `test.parquet`, plus a `dataset_card.json` capturing per-source counts before dedup and final class balance. An empty adversarial-eval template is created if none exists.

Final processed corpus: **21,963 examples** (14,642 benign / 7,321 injection) → 17,570 train / 2,196 val / 2,197 test.

## Training

`scripts/train_qlora.py` trains **Qwen/Qwen2-1.5B** as a 2-label sequence classifier:

- **Quantization**: 4-bit NF4 with double quantization, fp16 compute (`bitsandbytes`)
- **LoRA**: rank 8, alpha 16, dropout 0.1 targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`; classifier head trained fully (`SEQ_CLS` task type)
- **Optimization**: 3 epochs, effective batch size 32 (8 × grad-accum 4), lr 2e-4, cosine schedule, 3% warmup, weight decay 0.01, fp16, gradient checkpointing
- **Selection**: best checkpoint by validation **macro F1**, saved to `models/qwen-injection-detector/best`

## Calibration

Neural nets are overconfident, and this model was no exception. `scripts/calibrate.py`:

1. Collects validation logits once.
2. Grid-searches temperature over `{0.5 … 5.0}` by negative log-likelihood, then refines with bounded scalar minimization (`scipy.optimize.minimize_scalar`).
3. Saves the scalar as `models/qwen-injection-detector/best/temperature.pt`; inference divides logits by T before softmax.

Result: optimal **T = 3.51**, ECE reduced **65.9%** (0.0058 → 0.0020), NLL 0.064 → 0.027. Plots before/after reliability diagrams to `eval/calibration_curve.png`.

## Results

Test set (n = 2,197), from `eval/comparison.json`:

| Metric | Baseline (`deepset/deberta-v3-base-injection`) | Fine-tuned Qwen2-1.5B QLoRA | Δ |
|---|---|---|---|
| Accuracy | 0.6636 | **0.9873** | ▲ 0.3237 |
| Macro F1 | 0.6623 | **0.9857** | ▲ 0.3234 |
| ROC-AUC | 0.7643 | **0.9979** | — |
| Benign recall | 0.5473 | **0.9918** | ▲ 0.4445 |
| Injection recall | 0.8942 | **0.9783** | ▲ 0.0841 |
| Injection F1 | 0.6407 | **0.9810** | ▲ 0.3403 |

The baseline collapses on out-of-distribution benign data (26.96% accuracy on the `no_robots` source), which is exactly the failure mode fine-tuning fixes.

### Adversarial Evaluation

47 hand-crafted examples spanning 12 categories (direct injection, DAN/roleplay, hypothetical framing, Base64/ROT13/leetspeak, multi-language FR/ZH/HI/KO, system-prompt override, payload splitting, few-shot poisoning, benign lookalikes, zero-width/Unicode tricks, structured-data payloads, token smuggling). Run through the live middleware (`eval_adversarial.py`):

| Metric | Value |
|---|---|
| Accuracy | 0.8936 |
| Injection recall | 0.9535 |
| Benign recall (lookalikes) | 0.25 |
| Perfect categories | direct injection, DAN, Base64, multi-lang, special chars, structured data, token smuggling, few-shot |

Known weakness: over-blocking benign prompts that merely *mention* injection keywords — the motivation for soft-flag default mode and the review queue below.

## Serving Middleware

`middleware/app.py` exposes three endpoints:

- `POST /chat/completions` — OpenAI-compatible proxy. Concatenates all message contents, classifies them, then either blocks or forwards.
- `GET /health` — model readiness, active mode, threshold.
- `GET /metrics` — Prometheus metrics.

**Enforcement modes** (`MODE` setting):
- `hard_block` — returns `403` with `{"error": "Prompt injection detected", "confidence": ..., "injection_detected": true}`
- `soft_flag` (default) — forwards the request but attaches `X-Injection-Detected: <confidence>` response header

A request is treated as injection only when `label == 1` **and** calibrated confidence ≥ `THRESHOLD` (default 0.85).

**Shadow mode** (`SHADOW_ENABLED=true`): loads a second classifier (typically the newly retrained candidate promoted by `retrain.py --shadow`). Every request runs through both models; agreement/disagreement counts and absolute confidence deltas are exported as metrics, so a new model can be validated on live traffic without risk.

**Prometheus metrics** exposed at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `injection_requests_total` | counter | Total requests screened |
| `injection_blocked_total` | counter | Hard-blocked requests |
| `injection_flagged_total` | counter | Soft-flagged requests |
| `injection_confidence` | histogram | Confidence score distribution |
| `injection_latency_seconds` | histogram | Classifier latency |
| `injection_shadow_agreements_total` | counter | Shadow/prod label agreement |
| `injection_shadow_disagreements_total` | counter | Shadow/prod label disagreement |
| `injection_shadow_confidence_delta` | histogram | \|prod − shadow\| confidence gap |

### Configuration (environment variables or `.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_ENDPOINT` | `https://api.openai.com/v1` | Upstream OpenAI-compatible API |
| `LLM_API_KEY` | *(empty)* | Bearer key forwarded upstream |
| `THRESHOLD` | `0.85` | Confidence cutoff for block/flag |
| `MODE` | `soft_flag` | `soft_flag` or `hard_block` |
| `MODEL_PATH` | `models/qwen-injection-detector/best` | LoRA adapter directory |
| `MAX_LENGTH` | `512` | Tokenizer truncation length |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Server bind address |
| `SHADOW_ENABLED` | `false` | Enable parallel shadow classifier |
| `SHADOW_MODEL_PATH` | *(empty)* | Adapter dir for shadow model |
| `REVIEW_QUEUE_PATH` | `data/review_queue.db` | SQLite queue location |

## Human Review & Retraining Loop

1. Every proxied request (with production and shadow predictions) lands in a SQLite WAL-backed **review queue** (`scripts/review_queue.py`).
2. Label pending items interactively:
   ```bash
   python scripts/review_queue.py          # i=injection, b=benign, s=skip, q=quit
   python scripts/review_queue.py --stats  # queue statistics
   python scripts/review_queue.py --export data/reviewed_labels.jsonl
   ```
3. Close the loop:
   ```bash
   python scripts/retrain.py --shadow
   ```
   This merges human-labeled examples back into the corpus, dedups/rebalances/re-splits (originals backed up with timestamps), reruns training + calibration, optionally copies the result to the **shadow** slot, and writes `models/qwen-injection-detector/retrain_summary.json`. If training fails, the previous splits are restored automatically.

## Monitoring Stack

```bash
docker compose up -d
```

- **Prometheus** on `http://localhost:9090` — scrapes the middleware every 5s via `host.docker.internal:8080`, 7-day retention
- **Grafana** on `http://localhost:3000` (anonymous admin auth for local dev) — datasource and "Injection Detector" dashboard auto-provisioned from `configs/grafana/`

## Getting Started

Requires Python 3.10+ and a CUDA GPU for training/inference (falls back to CPU).

```bash
# 1. Training dependencies
pip install -r requirements.txt

# 2. Build the dataset
python scripts/prepare_dataset.py --config configs/dataset_config.yaml
python scripts/analyze_dataset.py    # sanity-check per-source stats

# 3. Baseline + training + calibration
python scripts/evaluate_baseline.py
python scripts/train_qlora.py --config configs/training_config.yaml
python scripts/calibrate.py

# 4. Evaluate on test set
python scripts/evaluate_model.py

# 5. Serve the detector
pip install -r middleware/requirements.txt
python -m middleware.app

# 6. Adversarial eval against the live server
python scripts/build_adversarial_eval.py
python scripts/eval_adversarial.py

# 7. Optional monitoring stack
docker compose up -d
```

### Client Integration

Point any OpenAI SDK client at the middleware — no other code changes needed:

```python
from openai import OpenAI
from middleware import wrap_openai_client

client = wrap_openai_client(
    OpenAI(api_key="..."),
    endpoint="http://localhost:8080",
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
if "X-Injection-Detected" in getattr(response, "headers", {}):
    ...  # flagged in soft_flag mode
```

Hard-blocked requests raise a `403` API error carrying `"Prompt injection detected"`.

### Tests

```bash
pip install pytest httpx
python -m pytest middleware/test_middleware.py -v
```

Covers health/metrics endpoints, required metric names, empty-message validation (400), benign passthrough, and hard-block behavior with confidence payload.

## Notes

- Model checkpoints and `.safetensors` binaries are gitignored; `models/qwen-injection-detector/best/` must be regenerated via the training pipeline (or restored separately) for the middleware to start.
- `convert_safetensors_to_bin.py` is a one-off Windows workaround converting the cached HF safetensors snapshot to `.bin` to avoid page-file issues with memory-mapped loading.
- The middleware loads the 4-bit quantized base model at startup and warms up with 3 dummy inferences before reporting ready; requests received earlier get a `503`.
