# Phase 5: Shadow Mode, Review Queue & Retrain Pipeline

## Overview

Three components work together to create a continuous improvement loop:

1. **Shadow Mode** — a second model runs in parallel with the production model. Its predictions are logged but never acted upon.
2. **Review Queue** — every request (plus shadow predictions) is stored in SQLite. A CLI lets human reviewers label borderline cases.
3. **Retrain Pipeline** — exports reviewed labels, merges them back into the training set, re-deduplicates, re-splits, and re-trains.

## Why Shadow Mode?

- Safe path for model upgrades: deploy a candidate model in shadow, compare its predictions to production over thousands of requests, and only promote it once agreement metrics look good.
- Detects data drift: if production and shadow diverge over time, that signals the distribution has shifted and retraining is needed.
- No user-facing risk: shadow predictions never affect the response the user sees.

## Architecture

```
                        +-----------------------------+
                        |        FastAPI App          |
                        |   /chat/completions          |
                        +--------+--------------------+
                                 |
                          +------+------+
                          |  Classifier |  (production)
                          +------+------+
                                 |
                          +------+------+
                          |    Shadow   |  (candidate model, optional)
                          +------+------+
                                 |
                          +------+------+
                          | ReviewQueue |  (SQLite: every request)
                          +------+------+
                                 |
                          +------+------+
                          |  CLI review  |  (human labels borderline)
                          +------+------+
                                 |
                          +------+------+
                          |   retrain   |  (merge + train + calibrate)
                          +-------------+
```

## Enabling Shadow Mode

Set environment variables or add to `.env`:

```bash
SHADOW_ENABLED=true
SHADOW_MODEL_PATH=models/qwen-injection-detector/shadow
```

When `SHADOW_ENABLED=true` and `SHADOW_MODEL_PATH` points to a valid LoRA adapter directory, the middleware loads a second `InjectionClassifier` instance during startup.

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `injection_shadow_agreements_total` | Counter | Production and shadow agreed on label |
| `injection_shadow_disagreements_total` | Counter | Production and shadow disagreed |
| `injection_shadow_confidence_delta` | Histogram | Absolute confidence difference between the two models |

A spike in disagreements is a strong signal to review the shadow model's performance or initiate retraining.

## Review Queue CLI

```bash
# Show queue statistics
python scripts/review_queue.py --db data/review_queue.db --stats

# Review pending items (interactive)
python scripts/review_queue.py --db data/review_queue.db --limit 20

# Export reviewed labels for retraining
python scripts/review_queue.py --db data/review_queue.db --export data/reviewed_labels.jsonl
```

The CLI shows each request with both production and shadow predictions, then prompts for a human label: `i` (injection), `b` (benign), `s` (skip), `q` (quit).

## Retrain Pipeline

```bash
# Full retrain: merge labels + dedup + split + train + calibrate + copy shadow
python scripts/retrain.py --shadow

# Data preparation only (no training)
python scripts/retrain.py --skip-train

# Custom paths
python scripts/retrain.py --db data/custom_queue.db --train-config configs/training_config.yaml
```

### What the pipeline does:

1. Exports all reviewed labels from the SQLite queue to `data/processed/reviewed_labels.jsonl`
2. Loads existing `train.parquet` and `val.parquet`
3. Appends reviewed labels with `source = "human_review"`
4. Re-runs exact + near deduplication (0.92 cosine threshold)
5. Re-balances to ~2:1 benign:injection
6. Stratified 80/10/10 split (backups created with timestamps)
7. Runs `train_qlora.py` with the updated splits
8. Runs `calibrate.py` to find the optimal temperature
9. If `--shadow`, copies the new best model to `models/qwen-injection-detector/shadow/`

On failure, the original data splits are restored from backups.

## Practical Considerations

- **GPU memory**: The shadow model is loaded on the same GPU. With QLoRA 4-bit + LoRA rank 8, two instances use ~3.2 GB × 2 = 6.4 GB (filling a 6.4 GB RTX 4050). If VRAM is tight, consider running shadow inference on CPU or disabling shadow on high-traffic periods.
- **Review frequency**: In early deployment, review 50–100 requests per day until confidence stabilizes. After that, weekly spot-checks suffice.
- **Cold start**: Both models run warm-up on startup (3 dummy calls + `cuda.synchronize()`), which takes ~2 seconds.
- **Retrain window**: A full retrain (3 epochs on ~18K samples) takes ~4 hours on RTX 4050. Plan retraining during low-traffic windows.
