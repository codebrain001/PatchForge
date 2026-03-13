# PatchForge

**From Photo to Patch** — an agentic AI pipeline that turns a photo or video of a broken object into a 3D-printable repair part.

## How It Works

1. **Upload** a photo or short video of the broken object
2. PatchForge **detects the damage**, **segments** it, **measures** it, and **infers the thickness** — all automatically
3. **Download** the generated STL patch or **send it directly** to a Bambu Lab A1 printer

Optional: upload a "before" photo for fully automatic damage detection — no clicking needed.

## Architecture

PatchForge is built as a **7-agent pipeline** where each agent executes a domain-specific function, reasons about its output using an LLM (structured JSON), and reports confidence scores visible in the UI.

```
Upload → Calibration → Segmentation → Measurement → Thickness → Mesh → Validation → Print
```

| Agent | Pipeline Functions | LLM Decision Role |
|-------|-------------------|----------|
| **Calibration** | ArUco / HEIF depth / reference line / WebXR / vision | Runs ALL strategies, sees ALL results, picks the best scale factor or blends them |
| **Segmentation** | SAM 2 with point prompts | Assesses mask quality, suggests better click points |
| **Measurement** | OpenCV contour analysis + vision measurement | Arbitrates between pixel-based and vision-based measurements |
| **Thickness** | LiDAR / Video MVS / side photo / monocular depth / vision | Runs ALL strategies, sees ALL estimates, decides final thickness |
| **Mesh** | Shapely + trimesh extrusion + chamfer | Assesses printability, suggests thickness/chamfer adjustments |
| **Validation** | Cross-checks all pipeline results | Final go/no-go judgment on the entire pipeline |
| **Printer** | bambulabs_api MQTT/FTP | Monitors print status, flags issues |

All agents extend a common base class (`app/agents/base.py`) with structured LLM output and an `AgentResult` containing reasoning, suggestions, confidence, and a proceed flag.

**The LLM is the decision engine** — every agent uses the LLM to arbitrate between competing strategies, assess confidence, and decide whether the pipeline should proceed. When multiple calibration or thickness estimates are available, the LLM runs a consensus protocol: it sees ALL candidates and picks the most trustworthy result (or blends them). This is not optional decoration — it is the core intelligence of the system.

### Consensus Decision Protocol

Calibration and thickness estimation both follow the same pattern:

```
┌──────────────────────────────────────────────────────────┐
│                   Run ALL strategies                     │
│                                                          │
│  HEIF depth ──┐                                         │
│  ArUco marker ─┼─── Collect ALL results ─── LLM sees    │
│  WebXR AR ─────┤    with confidence scores    every      │
│  Reference ────┘                            candidate    │
│                                                 │        │
│                              ┌───────────────────┘       │
│                              ▼                           │
│                   LLM Consensus Decision                 │
│                   ─ Pick best result                     │
│                   ─ Blend if strategies agree            │
│                   ─ Explain reasoning                    │
│                   ─ Flag implausible values              │
│                              │                           │
│                              ▼                           │
│                 Single authoritative result               │
│                 with explained confidence                 │
└──────────────────────────────────────────────────────────┘
```

The depth map extracted during calibration is passed directly to the thickness agent, avoiding redundant I/O and enabling cross-stage reasoning.

## Tech Stack

| Layer | Tool |
|-------|------|
| Web framework | FastAPI |
| Segmentation | SAM 2 (HuggingFace `transformers`) |
| Depth estimation | Depth Anything V2 |
| LLM reasoning | Google Gemini / OpenAI (configurable) |
| Computer vision | OpenCV |
| Geometry | Shapely + trimesh |
| 3D printer | bambulabs_api |
| 3D preview | Three.js |
| Persistence | SQLite |
| Deployment | Docker + NVIDIA Container Toolkit |

## Setup

### Prerequisites

- Python 3.10+ (recommended 3.12+)
- CUDA-capable NVIDIA GPU (or set `DEVICE=cpu` in `.env`)
- ~5 GB disk for model downloads on first run

### Install

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # macOS / Linux

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# Edit .env — set GEMINI_API_KEY at minimum

# 5. Run
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**.

### Docker

```bash
copy .env.example .env
# Edit .env with your API key
docker compose up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key ([get one](https://aistudio.google.com/apikey)) |
| `OPENAI_API_KEY` | — | OpenAI API key (optional fallback) |
| `LLM_PROVIDER` | `auto` | `gemini`, `openai`, or `auto` |
| `DEVICE` | auto | `cuda` or `cpu` |
| `SAM2_MODEL_NAME` | `facebook/sam2.1-hiera-small` | SAM 2 model |
| `DEPTH_MODEL_NAME` | `depth-anything/Depth-Anything-V2-Small-hf` | Depth model |

See `.env.example` for the full list including video, calibration, and printer settings.

## Project Structure

```
app/
├── main.py                  # FastAPI entry point
├── config.py                # Pydantic settings
├── agents/                  # 7 AI agents (base, orchestrator, domain agents)
├── pipeline/                # CV & geometry functions (calibration, segmentation, mesh, etc.)
├── api/                     # REST + WebSocket endpoints
├── models/                  # Job model, enums, API schemas
├── core/                    # Job store (SQLite), storage, LLM client, exceptions
└── static/
    └── index.html           # Guided 5-step wizard UI
```

## License

MIT
