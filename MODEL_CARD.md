# Model Card: Prompt Injection Detector

## Model Details

- **Base model**: Qwen2-1.5B (1.5B parameters)
- **Fine-tuning method**: QLoRA (4-bit NF4 quantization, rank 8)
- **Task**: Sequence classification (2 labels: benign, injection)
- **Trainable parameters**: 2.18M / 890M (0.24%)
- **Temperature scaling**: Learned T = 3.5104
- **Hardware**: NVIDIA RTX 4050 Laptop GPU (6.4 GB VRAM)
- **Training time**: ~4 hours (3 epochs)

## Intended Use

Real-time guardrail to detect prompt injection attacks in LLM application requests. Designed as a FastAPI middleware layer that inspects incoming messages before forwarding to the upstream LLM.

### Use Cases

- Blocking jailbreak attempts before they reach the LLM
- Flagging suspicious requests for human review (shadow mode)
- Monitoring injection attempt rates over time via Prometheus/Grafana

### Out-of-Scope

- Generative text classification or content filtering
- Detection of indirect prompt injection (retrieval-augmented generation context poisoning)
- Multi-modal inputs (images, audio)
- Production use without calibration and shadow-mode validation

## Training Data

| Source | Role | Samples (after dedup) |
|--------|------|----------------------|
| deepset/prompt-injections | Injection + benign | ~3,800 |
| S-Labs/prompt-injection-dataset | Injection + benign | ~4,200 |
| xTRam1/safe-guard-prompt-injection | Injection only | ~2,500 |
| Lakera/gandalf_ignore_instructions | Injection only | 350 |
| HuggingFaceH4/no_robots | Benign only | ~13,000 |

**Preprocessing**:
- Exact deduplication (case-insensitive, whitespace-normalized)
- Near-dedup at 0.92 cosine similarity (all-MiniLM-L6-v2)
- Balanced to ~2:1 benign:injection ratio
- Stratified 80/10/10 split with source-based stratification

**Final counts**: train=17,570, val=2,196, test=2,197

## Evaluation Results

### Test Set (n=2,197)

| Metric | Value |
|--------|-------|
| Accuracy | 98.73% |
| Macro F1 | 0.9857 |
| Injection Recall | 97.83% |
| Benign Recall | 99.18% |
| ROC AUC | 0.9986 |

### Calibration

| Metric | Before (T=1.0) | After (T=3.51) |
|--------|----------------|-----------------|
| ECE | 0.0284 | 0.0115 |
| NLL | 0.0685 | 0.0432 |

### Baseline Comparison (deberta-v3-base-injection, zero-shot)

| Metric | Zero-Shot | Fine-tuned (ours) |
|--------|-----------|-------------------|
| Accuracy | 66.36% | 98.73% |
| Macro F1 | 0.6623 | 0.9857 |
| Injection Recall | 89.42% | 97.83% |
| Benign Recall | 54.73% | 99.18% |

### Limitations

- **Adversarial robustness**: Quantitative adversarial evaluation pending (blocked by Windows page file fragmentation on training machine). The adversarial eval set covers 47 examples across 12 attack categories.
- **no_robots collapse**: The zero-shot baseline achieves only 27% accuracy on the no_robots subset (all benign, classified as injection). The fine-tuned model resolves this.
- **Indirect injection**: Not evaluated. The model inspects user messages only, not retrieved context.
- **Domain shift**: Performance on out-of-distribution prompts (different LLM, different task domains) is unknown.

## Bias & Fairness

- Training data is English-only (primarily LLM chat formats)
- The no_robots subset provides diverse instruction-following bening examples, but coverage of non-English or code-switching inputs is minimal
- Adversarial examples are English-only — multilingual injection attacks likely have higher miss rates

## Deployment Recommendations

1. Start in **shadow mode** (`SHADOW_ENABLED=true`) with a human review process for at least 500 requests
2. Validate on your own traffic distribution before enabling `hard_block` mode
3. Use calibrated confidence scores: treat predictions with confidence 0.5–0.8 as low-confidence and route to review
4. Retrain monthly or after 5,000+ human-labeled reviews are collected

## Environmental Impact

- **Training emissions**: ~0.4 kg CO2e (estimated, 4 hours on RTX 4050 at ~65W)
- **Inference**: ~3.2 GB VRAM, ~35 ms P50 latency on RTX 4050
- **Quantization**: 4-bit NF4 reduces model size from ~3 GB to ~0.9 GB for the base model

## Citation

```bibtex
@software{prompt_injection_detector_2025,
  author = {Yash Desai},
  title = {Prompt Injection Detector: QLoRA Fine-tuned Qwen2-1.5B},
  year = {2025},
  url = {https://github.com/yash/prompt-injection-detector}
}
```
