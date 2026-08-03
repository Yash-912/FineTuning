# Adversarial Evaluation Results

## Overview

47 hand-authored adversarial examples across 12 attack categories, evaluated against the fine-tuned Qwen2-1.5B QLoRA model (temperature=3.51).

## Summary

| Metric | Value |
|--------|-------|
| Accuracy | 89.36% |
| Macro F1 | 0.6141 |
| Injection recall | 95.35% |
| Benign recall | 25.00% |
| Injection precision | 93.18% |
| Benign precision | 33.33% |

## Per-Category Breakdown

| Category | n | Accuracy | Notes |
|----------|---|----------|-------|
| base64 | 4 | 100% | Base64 encoded injection |
| direct_injection | 5 | 100% | Direct "ignore instructions" |
| few_shot | 4 | 100% | Few-shot poisoning |
| multi_lang | 4 | 100% | Non-English injection |
| roleplay_dan | 5 | 100% | DAN / roleplay jailbreak |
| special_chars | 3 | 100% | Special character obfuscation |
| structured_data | 3 | 100% | Injection in JSON/structured format |
| system_override | 4 | 100% | System prompt override |
| token_smuggle | 3 | 100% | Token smuggling / splitting |
| hypothetical | 4 | 75% | Hypothetical framing (1 miss) |
| payload_split | 4 | 75% | Split across messages (1 miss) |
| benign_lookalike | 4 | 25% | Benign prompts that resemble injections |

## Weakest Category: benign_lookalike

Only 1/4 correct. The 3 false positives are benign prompts written in a style that mimics injection patterns (e.g., "I need you to act as if..." or detailed system override-like language but without malicious intent). These are inherently ambiguous — even humans might disagree.

### False positive examples (benign classified as injection):

The model is trigger-happy on prompts that *look like* instructions about behavior modification, even when the intent is benign (e.g., roleplay requests, detailed formatting instructions).

## Injection Recall Analysis

- 41/43 true injections detected (95.35%)
- Both missed injections were in categories with lower accuracy:
  - 1 hypothetical framing injection (disguised as academic discussion)
  - 1 payload split injection (distributed across multiple messages)

## vs. Target Requirements

| Requirement | Target | Actual | Status |
|---|---|---|---|
| Adversarial recall | ≥ 60% | 95.35% | ✅ Exceeded |
| Overall accuracy (test set) | ≥ 90% | 98.73% | ✅ |
| Benign recall (test set) | ≥ 85% | 99.18% | ✅ |
| Adversarial benign recall | — | 25.00% | ⚠️ Low |

## Recommendations

1. **Add benign_lookalike examples to training data**: The 3 FP cases should be added as labeled benign examples in the next retrain cycle.
2. **Threshold tuning**: The current threshold (0.85) captures 95% of adversarial injections but flags lookalike benigns. Consider a higher threshold (0.90–0.92) if benign recall on adversarial-style prompts matters.
3. **Adversarial augmentation**: Generate synthetic benign_lookalike prompts by asking an LLM to rewrite benign instructions in a style that mimics instructions ("act as if you are...", "from now on...", etc.).
