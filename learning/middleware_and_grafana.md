# Phases 3 & 4: FastAPI Middleware & Grafana Monitoring

## Overview

We built an HTTP middleware layer that sits between any OpenAI-compatible LLM
client and the real LLM endpoint. Every request passes through our detector
before being forwarded. We also added Prometheus + Grafana for observability.

### Architecture Diagram

```
                    +------------------------------------------+
                    |         Your Application                 |
                    |    client = OpenAI(api_key=...)          |
                    |    client = wrap_openai_client(          |
                    |        client, "http://localhost:8080")  |
                    +---------------------+--------------------+
                                          | POST /chat/completions
                                          v
+-------------------------------------------------------------+
|              FastAPI Middleware (port 8080)                  |
|  1. Parse incoming request (OpenAI format)                  |
|  2. Extract text from messages                              |
|  3. Classify with Qwen2-1.5B QLoRA model                    |
|  4. If injection + threshold -> hard-block (403) or flag    |
|  5. Forward to real LLM (e.g. api.openai.com)               |
|  6. Record Prometheus metrics                               |
+------------------+---------------------+--------------------+
                   |                     | GET /metrics
                   | POST /chat/completions (forwarded)
                   v                     v
+---------------------+   +-------------------------+
|   Real LLM Endpoint |   |  Prometheus (9090)      |
| (OpenAI / any)      |   |  Grafana (3000)         |
+---------------------+   +-------------------------+
```

## Phase 3: The Middleware (7 files in `middleware/`)

### 3A. Configuration - `middleware/config.py`

Uses **Pydantic BaseSettings** so every config value can be set via
environment variables or a `.env` file.

| Field | Default | Description |
|---|---|---|
| `llm_endpoint` | `https://api.openai.com/v1` | Real LLM to forward requests to |
| `llm_api_key` | `""` | API key for the LLM (from env var) |
| `threshold` | `0.85` | Minimum confidence to trigger block/flag |
| `mode` | `soft_flag` | `"hard_block"` or `"soft_flag"` |
| `model_path` | `models/qwen-injection-detector/best` | Path to saved LoRA adapters |
| `max_length` | `512` | Max token length for classification |
| `host` | `0.0.0.0` | Bind address |
| `port` | `8080` | Server port |

How to set config via env:
```bash
$env:MODE = "hard_block"
$env:THRESHOLD = "0.5"
python -m middleware.app
```

Or create a `.env` file in the project root:
```ini
MODE=hard_block
THRESHOLD=0.5
```

### 3B. The Classifier - `middleware/classifier.py` (Singleton Pattern)

**Why Singleton?** PyTorch model loading takes ~30 seconds and 2GB+ VRAM.
We must load it exactly once and share it across all requests.

The class implements the Singleton pattern via `__new__`:
```python
class InjectionClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path, max_length):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()
        self._warmup()
```

**Loading sequence in `_load_model()`:**
1. Load tokenizer (`Qwen/Qwen2-1.5B`)
2. Set `pad_token = eos_token` (Qwen2 has no pad token by default)
3. Configure 4-bit NF4 quantization (`BitsAndBytesConfig`)
4. Load base model via `AutoModelForSequenceClassification`
5. Load LoRA adapters via `PeftModel.from_pretrained`
6. Load temperature from `temperature.pt`

**Why NO merge_and_unload?** On a 6GB RTX 4050, merging the LoRA
adapters into the base model exceeds VRAM. We keep them separate.

**Warm-up (`_warmup()`):** 3 dummy inferences + `torch.cuda.synchronize()`
to trigger CUDA kernel compilation. Without warm-up, the first real
request has a 1-2 second cold-start latency.

**The predict method:**
```python
@torch.no_grad()
def predict(self, text: str) -> tuple[int, float]:
    inputs = self.tokenizer(text, truncation=True, max_length=512)
    logits = self.model(**inputs).logits
    logits = logits.float()       # bfloat16 -> float32 for numpy
    probs = softmax(logits / temperature, dim=-1)
    predicted_class = argmax(probs)
    confidence = probs[predicted_class]
    return predicted_class, confidence
```

The `logits.float()` call is critical - PyTorch on CUDA defaults to
bfloat16, but numpy doesn't support it.

### 3C. The FastAPI Server - `middleware/app.py`

Three endpoints:

**`GET /health`** - Returns model status, mode, and threshold.
```json
{"status":"ok","model_loaded":true,"mode":"hard_block","threshold":0.5}
```

**`GET /metrics`** - Returns Prometheus-formatted text from a private
`CollectorRegistry` (avoids "Duplicated timeseries" errors).

**`POST /chat/completions`** - The main proxy endpoint:
1. Parse JSON body (OpenAI chat format)
2. `extract_text(messages)` - concatenates all message contents
3. `classifier.predict(text)` - get (label, confidence)
4. Record Prometheus metrics
5. If injection detected above threshold:
   - `hard_block` mode -> return 403
   - `soft_flag` mode -> forward but add `X-Injection-Detected` header
6. Forward to real LLM via `httpx.AsyncClient`
7. Return the real LLM response

**The `lifespan` async context manager** handles startup and shutdown:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = InjectionClassifier(...)
    app.state.classifier = classifier
    app.state.httpx_client = httpx.AsyncClient(timeout=60.0)
    yield
    await app.state.httpx_client.aclose()
```

### 3D. One-Line Integration - `middleware/__init__.py`

```python
# Before (no protection):
client = OpenAI(api_key="sk-...")

# After (one-line guardrail):
client = wrap_openai_client(OpenAI(api_key="sk-..."), endpoint="http://localhost:8080")
```

Sets `client.base_url` to point at our middleware instead of OpenAI.

### 3E. Thread Safety

PyTorch's C++ backend releases the Python GIL during `model.forward()`.
Multiple threads can call `predict()` simultaneously - they queue on the
GPU without blocking each other in Python.

### 3F. Test File - `test_middleware.py`

Uses FastAPI's `TestClient`:
- Health and metrics return 200
- Empty body returns 400
- Known injection triggers 403 in hard_block mode
- Known benign text passes through

## Phase 4: Observability (5 files)

### 4A. `configs/prometheus.yml`

Scrapes `host.docker.internal:8080/metrics` every 5s.

### 4B. `configs/grafana/datasources/prometheus.yml`

Auto-provisions Prometheus datasource at `http://prometheus:9090`.

### 4C. `configs/grafana/dashboards/dashboard.yaml`

Auto-provisions dashboards from JSON files. Updates every 10s.

### 4D. `configs/grafana/dashboards/injection_detector.json`

6 panels:
| Panel | Query | Purpose |
|---|---|---|
| Request Rate | `rate(injection_requests_total[1m])` | Requests/sec |
| Blocked vs Flagged | `rate(injection_blocked_total[1m])` + `rate(injection_flagged_total[1m])` | Detection rate |
| Confidence Heatmap | `sum(rate(injection_confidence_bucket[1m])) by (le)` | Confidence distribution |
| Latency P50/P99 | `histogram_quantile(0.50/0.99, rate(...))` | Latency with thresholds |
| Injection Rate | `(blocked + flagged) / total` | % detected |
| Model Status | `up{job="injection-detector"}` | UP/DOWN |

### 4E. `docker-compose.yml`

Prometheus (9090) + Grafana (3000), anonymous admin access.

## How to Run

```bash
# Terminal 1: middleware
$env:MODE = "soft_flag"
$env:THRESHOLD = "0.85"
venv\Scripts\python -m middleware.app

# Terminal 2: monitoring
docker compose up -d

# Open http://localhost:3000 (no login)
```

## Prometheus Metrics

| Metric | Type | Purpose |
|---|---|---|
| `injection_requests_total` | Counter | Total requests |
| `injection_blocked_total` | Counter | Hard-blocked (403) |
| `injection_flagged_total` | Counter | Soft-flagged (header) |
| `injection_confidence` | Histogram | Confidence distribution |
| `injection_latency_seconds` | Histogram | Latency distribution |

## Gotchas

1. **Windows page file error (os error 1455):** safetensors memory-mapping
   can crash. Convert to `.bin` with `scripts/convert_safetensors_to_bin.py`.

2. **Prometheus "Duplicated timeseries":** Use private `CollectorRegistry()`.

3. **bfloat16 -> numpy:** `logits.float()` before `.numpy()`.

4. **Qwen2 pad token:** `tokenizer.pad_token = tokenizer.eos_token`.

5. **CUDA cold start:** 3 warm-up calls with `torch.cuda.synchronize()`.
