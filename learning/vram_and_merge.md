# VRAM, Precision, and the merge_and_unload Trap

## Why Inference Was 5+ Minutes (And How to Never Do It Again)

### The 30-Second Story

Our training used **4-bit quantization** (NF4) — the base model weights were stored as 4-bit integers, using ~0.75GB. This fit comfortably in 6GB VRAM alongside LoRA adapters, gradients, and activations.

But in `evaluate_model.py`, we loaded the base model with `dtype=torch.float16` — **no quantization**. The base model alone became ~3GB. Then when `merge_and_unload()` merged the LoRA fp32 adapters into the base, PyTorch upcasted everything to **fp32** — the model became either ~6GB (fp32 full model) or had mixed precision issues. Add activations for batch_size=64 and sequence_length=512, and you blow past 6.4GB VRAM.

The GPU runs out of memory → CUDA starts **paging to system RAM** → each memory access goes from nanoseconds to microseconds → your 60-second job takes 5+ minutes.

### The Full Explanation

#### 1. What Is a Model's "Size" in Memory?

Every parameter in a neural network is a number. The number of **bits** used to store that number determines the precision and the memory footprint.

| Format | Bits per param | Memory for 1.5B params |
|---|---|---|
| float32 (fp32) | 32 | 6.0 GB |
| float16 (fp16) | 16 | 3.0 GB |
| int8 | 8 | 1.5 GB |
| **nf4** (4-bit) | **4** | **0.75 GB** |

This is just the weights. During inference you also need:
- **Activations** (intermediate layer outputs): depends on batch_size × seq_length × hidden_dim
- **KV cache** (if doing generation, not relevant for classification)

For a batch of 64 sequences of length 512:
- Each activation tensor is 64 × 512 × 2048 ≈ 67M floats ≈ 256MB in fp16
- ~24 layers × 256MB = **~6GB for activations alone** in fp16

**Total VRAM needed with fp16 + no quantization:**
```
Weights:    3.0 GB (fp16)
Activations: ~6.0 GB (fp16, batch=64)
--------------------
Total:       ~9.0 GB → exceeds 6.4 GB
```

**Total VRAM needed with 4-bit quantization:**
```
Weights:    0.75 GB (nf4, dequantized on-the-fly)
Activations: ~2.0 GB (we'd use smaller batch due to memory)
--------------------
Total:       ~3.0 GB → fits comfortably
```

#### 2. The merge_and_unload() Trap

`merge_and_unload()` does:

```
W_merged = W_base + A × B
```

Where:
- `W_base` is the original weight matrix (e.g., 2048 × 2048)
- `A` is the LoRA down-projection (2048 × 8)
- `B` is the LoRA up-projection (8 × 2048)

The result `W_merged` is the same size as `W_base`. But here's the trap:

**If `W_base` is fp16 and `A` and `B` are fp32 (which LoRA adapters often are):**

PyTorch must upcast `W_base` to fp32 to do the addition, then the result stays in fp32. The entire model silently becomes fp32 — **doubling memory**.

Even if they're both the same dtype, the merged model is still the **full-size** matrix. There's no 4-bit advantage because the merge is happening at the precision you loaded the base with.

**The rule:** Only merge when you need zero-overhead inference AND you have enough VRAM for the full-precision model. For our 6GB constraint, NEVER merge — keep the 4-bit quantized model with LoRA adapters separate.

#### 3. Why Training Works But Inference Was Slow

| Phase | Base model | LoRA | Memory | Works? |
|---|---|---|---|---|
| **Training** | 4-bit NF4 (~0.75GB) | fp32 adapters (~17MB) | ~4-5GB total | ✅ |
| **Eval (broken)** | fp16 (~3GB) then merged to fp32 (~6GB) | Merged in | ~9GB+ total | ❌ Overflows |
| **Eval (fixed)** | 4-bit NF4 (~0.75GB) | fp32 adapters (~17MB) | ~2-3GB total | ✅ |

Training worked because the quantization config (`BitsAndBytesConfig`) kept the base in 4-bit. The eval scripts didn't use quantization at all — they loaded the full fp16 model.

#### 4. How to Correctly Load for Inference on a 6GB GPU

```python
# ❌ WRONG - will merge to fp32 and blow VRAM:
base = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, dtype=torch.float16
)
model = PeftModel.from_pretrained(base, checkpoint_dir)
model = model.merge_and_unload()  # BAD: upcasts to fp32

# ✅ CORRECT - keep 4-bit, don't merge:
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=bnb_config,  # Keep 4-bit!
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(model, checkpoint_dir)
# NO merge_and_unload() - adapters stay separate
model = model.to(device)
model.eval()
```

With this approach:
- Base model stays in 4-bit (~0.75GB)
- LoRA adapters stay as separate fp32 modules (~17MB)
- At inference, each forward pass: dequantize 4-bit → fp16 → add LoRA → compute
- Peak VRAM: ~2-3GB for batch_size=64 → well within 6GB

#### 5. The Mental Model: Always Track Your VRAM

Before running ANY inference or training, estimate:

```
total_vram = weights_memory + activation_memory + overhead

weights_memory = num_params × bytes_per_param
activation_memory ~ batch_size × seq_length × hidden_dim × num_layers × bytes_per_value
overhead = ~500MB for CUDA context, tokenizers, etc.
```

| Task | Estimate | Fits in 6GB? |
|---|---|---|
| 1.5B model, 4-bit, batch=64 | ~3GB | ✅ |
| 1.5B model, fp16, batch=64 | ~9GB | ❌ |
| 1.5B model, merged fp32, batch=64 | ~12GB | ❌❌ |
| 7B model, 4-bit, batch=1 | ~4GB | ✅ (just barely) |
| 7B model, fp16, batch=1 | ~14GB | ❌ |

**Always check VRAM before adding new components like `merge_and_unload()`.**

#### 6. When WOULD You Merge?

Merging is useful when:
- You have **abundant VRAM** (24GB+ GPUs like RTX 4090, A10, A100)
- You need **maximum inference throughput** (merging removes the LoRA addition overhead)
- You're deploying to **CPU** (merged model can be exported to ONNX)

For our 6GB RTX 4050: **never merge during inference**. Keep the 4-bit + separate adapters approach.

#### 7. Summary: One Concept, Three Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Loading in fp16 instead of 4-bit | 2-4x more VRAM than expected | Use `BitsAndBytesConfig` |
| Calling `merge_and_unload()` | Silent upcast to fp32, doubles VRAM | Don't merge on low-VRAM GPUs |
| Not estimating VRAM before running | Wait 5+ minutes for a 60-second job | Calculate before you execute |

**The golden rule:** If training used 4-bit quantization, inference should too — unless you have a specific reason and enough VRAM to do otherwise.
