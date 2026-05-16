# Continue HF Bridge

[![Continue](https://img.shields.io/badge/Continue-AI%20Coding%20Assistant-blue?logo=visual-studio-code)](https://continue.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces)

Bridge server connecting Continue IDE to free HuggingFace Spaces models via OpenAI-compatible API.

## Supported Models

| Model | HuggingFace Space | Adapter key |
|---|---|---|
| **GLM-4.5** (9B) | [zai-org/GLM-4.5-Space](https://huggingface.co/spaces/zai-org/GLM-4.5-Space) | `glm` |
| **Qwen3-VL-235B** (A22B MoE) | [Qwen/Qwen3-VL-235B-A22B-Instruct-Demo](https://huggingface.co/spaces/Qwen/Qwen3-VL-235B-A22B-Instruct-Demo) | `qwen` |

Adding a new model requires only implementing a small adapter class and registering it. See [Adding a New Adapter](#adding-a-new-adapter).

## How It Works

```
┌──────────┐   OpenAI / Ollama API   ┌───────────────────┐   Gradio SSE   ┌──────────────┐
│ Continue │ ───────────────────────▸ │  Continue Adapter  │ ─────────────▸ │  HF Space    │
│   IDE    │ ◂─────────────────────── │  (this server)     │ ◂───────────── │  (free GPU)  │
└──────────┘   streaming chunks       └───────────────────┘   SSE stream    └──────────────┘
```

1. **Continue** sends a standard `/v1/chat/completions` (or Ollama `/api/chat`) request.
2. The adapter selects the right backend based on the `model` field in the request.
3. The adapter translates the request into the Gradio queue API format, calls the HuggingFace Space, and streams the response back in OpenAI/Ollama chunk format.
4. Tool calls produced by the model are parsed, validated, and normalized before being returned to Continue.

### Key Features

- **Multi-model support** — select the model from Continue's UI; the adapter routes to the correct backend automatically.
- **Gradio SSE bridge** — translates between OpenAI streaming and Gradio's queue-based SSE protocol.
- **Tool call normalization** — fixes common model output issues:
  - Name aliases (`edit_file` → `edit_existing_file`)
  - Argument aliases (`path` → `filepath`, `content` → `contents`)
  - JSON unescape (converts `\\n` to real newlines in file contents)
  - Filepath sanitization (strips `./` prefixes, removes stray quotes)
  - Auto-converts `create_new_file` → `edit_existing_file` to avoid "already exists" errors
- **Chunked argument streaming** — tool call arguments are streamed in small pieces to prevent timeout errors.
- **Agent-mode system prompt** — built-in system prompts that guide models to produce well-structured JSON tool calls.

## Project Structure

```
continue_adapter/
├── run.py                          # Entry point
├── Pipfile                         # Python dependencies
├── app/
│   ├── main.py                     # FastAPI application factory
│   ├── config.py                   # Environment-based configuration
│   ├── adapters/
│   │   ├── base.py                 # Abstract ModelAdapter interface
│   │   ├── glm.py                  # GLM-4.5 Gradio adapter
│   │   ├── qwen.py                 # Qwen3-VL-235B Gradio adapter
│   │   └── registry.py             # Adapter registry + model routing
│   ├── routes/
│   │   ├── openai.py               # /v1/chat/completions, /v1/completions, /v1/models
│   │   └── ollama.py               # /api/chat, /api/generate, /api/tags
│   ├── services/
│   │   └── model_turn.py           # Orchestration: prompt → model → parse → respond
│   ├── parsing/
│   │   ├── clean.py                # Response cleanup
│   │   ├── json_repair.py          # Fuzzy JSON repair for malformed model output
│   │   ├── tool_calls.py           # Tool call extraction from various formats
│   │   └── transcript.py           # Conversation transcript builder
│   ├── tools/
│   │   ├── continue_defaults.py    # Default tool definitions and alias tables
│   │   ├── validation.py           # Tool call validation, fixing, and conversion
│   │   ├── intent.py               # Intent classification
│   │   └── registry.py             # Tool registry
│   └── schemas/
│       └── requests.py             # Pydantic request models
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Pipenv](https://pipenv.pypa.io/) (or install packages manually from `Pipfile`)

### Installation

```bash
git clone <repo-url> continue_adapter
cd continue_adapter
pipenv install
```

### Running

```bash
# Default: GLM-4.5 on port 11434
python run.py

# Use Qwen adapter by default
ADAPTER=qwen python run.py

# Custom port
PORT=8080 python run.py
```

### Configure Continue

Add the model(s) to your Continue config (`~/.continue/config.yaml`):

```yaml
models:
  - name: GLM-4-5 Local
    provider: openai
    model: glm-4-5
    apiBase: http://127.0.0.1:11434/v1
    apiKey: none

  - name: Qwen3-VL-235B
    provider: openai
    model: qwen3-vl-235b
    apiBase: http://127.0.0.1:11434/v1
    apiKey: none
```

Both models will be available in the Continue model picker. The adapter routes each request to the correct HuggingFace Space based on the `model` field.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ADAPTER` | `glm` | Default adapter when model name is unknown |
| `MODEL_NAME` | `glm-4-5` | Model name reported by `/v1/models` |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `11434` | Server port |
| `MAX_TRANSCRIPT_CHARS` | `14000` | Max conversation context sent to the model |
| `FALLBACK_MODE` | `false` | Skip remote model calls (for testing) |
| **GLM** | | |
| `GLM_ENDPOINT` | `https://zai-org-glm-4-5-space.hf.space/gradio_api` | GLM Space URL |
| `GLM_TIMEOUT_TOTAL` | `150` | Total timeout (seconds) |
| `GLM_TIMEOUT_PER_LINE` | `45` | Per-SSE-line timeout |
| `GLM_MAX_RETRIES` | `1` | Retry count on failure |
| **Qwen** | | |
| `QWEN_ENDPOINT` | `https://qwen-qwen3-vl-235b-a22b-instruct-demo.hf.space/gradio_api` | Qwen Space URL |
| `QWEN_TIMEOUT_TOTAL` | `180` | Total timeout (seconds) |
| `QWEN_TIMEOUT_PER_LINE` | `60` | Per-SSE-line timeout |
| `QWEN_ZEROGPU_TOKEN` | *(empty)* | ZeroGPU auth token (optional) |
| `QWEN_ZEROGPU_UUID` | *(empty)* | ZeroGPU session UUID (optional) |

## API Endpoints

### OpenAI-compatible

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `POST` | `/v1/completions` | FIM / tab-autocomplete |
| `GET` | `/v1/models` | List available models |

### Ollama-compatible

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Chat (streaming) |
| `POST` | `/api/generate` | Text generation |
| `GET` | `/api/tags` | List models |

## Adding a New Adapter

1. Create `app/adapters/my_model.py`:

```python
from app.adapters.base import ModelAdapter

class MyModelAdapter(ModelAdapter):
    async def call(self, prompt, *, retry_hint="", cancel_event=None, system_prompt=None):
        # Call your model API and return the raw text response
        ...
```

2. Register it in `app/adapters/registry.py`:

```python
_REGISTRY = {
    "glm": "app.adapters.glm:GLMAdapter",
    "qwen": "app.adapters.qwen:QwenAdapter",
    "my_model": "app.adapters.my_model:MyModelAdapter",  # ← add
}

_MODEL_TO_ADAPTER = {
    ...
    "my-model-name": "my_model",  # ← add
}
```

3. Run with `ADAPTER=my_model python run.py` or add to Continue config with `model: my-model-name`.

## Architecture Notes

- **One model call per request** — Continue handles the agent loop (tool execution, context injection). The adapter only performs a single model inference step per HTTP request.
- **Tool calls are parsed, not executed** — the adapter extracts tool calls from the model's text output and returns them in OpenAI function-calling format. Continue's IDE extension executes the actual tools.
- **Gradio protocol** — both adapters use HuggingFace's Gradio SSE queue API (`/queue/join` + `/queue/data`). The Qwen adapter uses a two-step flow: `/add_text` (append user message) → `/predict` (generate response).
- **Stateless** — no conversation state is stored server-side. Each request contains the full message history from Continue.

## License

MIT
