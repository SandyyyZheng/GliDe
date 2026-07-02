# 🪐 GliDe — Open-Ended Video Game Glitch Detection with Agentic Reasoning and Temporal Grounding

An agentic, LangGraph-based multimodal pipeline for automated video game glitch detection.

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/SandyZheng33/VideoGlitchBench)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2604.07818)

---

## 🏔️ Architecture

![Framework](figure/framework.jpg)

GliDe processes a video through five sequential stages:

- **Preprocess** — Extracts frames at a fixed FPS (default 4 fps) and stitches them into windows (default 8 frames per window) for downstream processing.
- **Scanner** — Runs a fast initial screening over every window to produce a glitch hypothesis (`has_glitch`, `category`, `confidence`) and a `game_context` description used as a RAG-like knowledge base by later stages.
- **Analyzer** — For windows flagged by the Scanner, runs an iterative investigation loop: a **Planner** selects the next tool, an **Executor** runs it, and a **Reflector** evaluates the result via an adversarial debate between an **Advocate** (game test engineer, argues for glitch), a **Skeptic** (game designer, argues for normal behavior), and a **Judge** (tech lead, makes the ruling).
- **Grounder** — Clusters analysis results across windows, merges adjacent occurrences of the same glitch, and performs bidirectional temporal boundary refinement.
- **Summarizer** — Converts grounded glitch records into the final report, translating frame indices to timestamps and using an LLM to produce clean, coherent descriptions.

---

## ☔ Tools

| Tool | Status | Description |
|------|--------|-------------|
| `vqa` | Active | Visual QA on the full stitched window image via MLLM |
| `zoom_in` | Active | Crop and magnify a region of interest, then run VQA |
| `object_tracking` | Optional | Frame-by-frame SAM3 tracking + automatic physics analysis (requires SAM3 installation) |

`object_tracking` is lazily initialized. SAM3 is only loaded on the first call, and the tool disables itself gracefully if SAM3 is not installed.

---

## 🧭 Installation

### 1. Install SAM3

`object_tracking` requires [SAM3](https://github.com/facebookresearch/sam3). Follow the official installation instructions in the SAM3 repository before proceeding. If you skip this step, the `object_tracking` tool will be automatically disabled and the rest of the pipeline will still work.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 🌪️ Quick Start

### Run with a local vLLM server

```bash
# Start vLLM first:
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000

python run.py --video data/videos/video_name.mp4
```

### Run with OpenAI

```bash
python run.py \
    --video data/videos/video_name.mp4 \
    --api-key $OPENAI_API_KEY \
    --api-base https://api.openai.com/v1 \
    --model gpt-4o \
    --game-name "GTA V"
```

### Batch processing

Two folder layouts are supported. Per-video reports and logs are written as usual; a consolidated `batch_report.json` is also saved.

**Benchmark layout** (`<GameName>/<video.mp4>` sub-folders, e.g. VideoGlitchBench) — game name is inferred automatically from each sub-folder name:

```bash
python run.py \
    --video-dir /path/to/VideoGlitchBench/ \
    --api-key $OPENAI_API_KEY \
    --api-base https://api.openai.com/v1 \
    --model gpt-4o
```

**Flat layout** (all videos directly in one folder) — supply a game name explicitly:

```bash
python run.py \
    --video-dir data/videos/ \
    --game-name "GTA V" \
    --api-key $OPENAI_API_KEY \
    --api-base https://api.openai.com/v1 \
    --model gpt-4o
```

---

## 🛸 Output

### Single video

The report is saved to `{output_dir}/results/{video_name}_report.json`:

```json
{
  "video_name": "haj831",
  "game_name": "GTA V",
  "no_bugs": false,
  "bugs": [
    "A red sports car is floating above the road surface near the highway overpass, with no visible support or propulsion."
  ],
  "time_nodes": [
    [[12, 15], [23, 24]]
  ]
}
```

`time_nodes[i]` is a list of `[start_sec, end_sec]` intervals for bug `i`.

### Batch

A consolidated report is saved to `{output_dir}/results/batch_report.json` as a JSON array of per-video reports. In benchmark layout each entry carries its own `game_name` derived from the sub-folder:

```json
[
  {
    "video_name": "clip_01",
    "game_name": "GTA V",
    "no_bugs": false,
    "bugs": ["..."],
    "time_nodes": [[[12, 15]]]
  },
  {
    "video_name": "clip_02",
    "game_name": "Black Mesa",
    "no_bugs": true,
    "bugs": [],
    "time_nodes": []
  }
]
```

---

## 🛰️ LangGraph Flow

GliDe uses [LangGraph](https://github.com/langchain-ai/langgraph)'s `StateGraph` to wire the pipeline together. Each stage is a **node** that reads from and writes to a shared `BugAgentState` TypedDict. State is passed immutably between nodes — each node returns only the keys it updates.

The edge from `scanner_node` is **conditional**: if no glitches were found, the graph skips directly to `summarizer_node`, avoiding unnecessary analyzer and grounder calls.

```
preprocess_node → scanner_node
                       │
                       ├── (has glitches) ──► analyzer_node ──► grounder_node ──► summarizer_node
                       │
                       └── (no glitches) ────────────────────────────────────► summarizer_node
```

---

## 🧱 Configuration Reference

```python
from config import BugAgentConfig

cfg = BugAgentConfig(
    output_dir="data",
    verbose=True,
    save_intermediate=True,   # saves scan/analysis/grounded JSONs to data/intermediate/
)

cfg.llm.api_key    = "EMPTY"
cfg.llm.api_base   = "http://localhost:8000/v1"
cfg.llm.model      = "Qwen/Qwen2.5-VL-7B-Instruct"
cfg.llm.temperature = 0.3
cfg.llm.max_tokens  = 1024
cfg.llm.timeout     = 120

cfg.preprocess.target_fps    = 4.0   # frames/sec to extract
cfg.preprocess.window_size   = 8     # frames per stitched window
cfg.preprocess.window_overlap = 0

cfg.scanner.temperature = 0.3
cfg.scanner.max_tokens  = 512

cfg.analyzer.max_iterations      = 5     # max Planner→Executor→Reflector cycles
cfg.analyzer.confidence_threshold = 0.80 # stop when Judge reaches this confidence
cfg.analyzer.sam3_gpus            = [1]  # GPU(s) for SAM3 (keep separate from the VLM GPU)

cfg.grounder.frames_per_window = 8  # must match preprocess.window_size

cfg.summarizer.fps = 4.0   # must match preprocess.target_fps
```

---

## ⚓ Evaluation

`groundtruth.json` is our manually annotated dataset, containing per-video ground-truth glitch descriptions and temporal intervals in the same format as the pipeline output.

Evaluation compares a `batch_report.json` against `groundtruth.json` using LLM-based description scoring (0–5) and temporal IoU, then reports precision, recall, and F1 in both raw and IoU-weighted forms.

### Option A — Local vLLM scoring model

```bash
# Start a scoring LLM (separate from the detection model):
CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --max-model-len 8192

python evaluation/run.py \
    --predictions data/results/batch_report.json \
    --groundtruth groundtruth.json \
    --api-base http://localhost:8001/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output data/results/eval.json
```

### Option B — OpenAI API scoring model

```bash
python evaluation/run.py \
    --predictions data/results/batch_report.json \
    --groundtruth groundtruth.json \
    --api-key $OPENAI_API_KEY \
    --api-base https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --output data/results/eval.json
```

`--output` is optional; if provided, per-video scores and match details are saved to the specified JSON file.

### Metrics

| Metric | Description |
|--------|-------------|
| `mean_score` | Average LLM description quality score (0–5) over matched pairs |
| `mean_iou` | Average temporal IoU over matched pairs |
| `precision / recall / f1` | Score-weighted detection metrics (max score = 5) |
| `precision_iou / recall_iou / f1_iou` | Same metrics further weighted by temporal IoU |

---

## 🌟 Citation

If you find this work useful, please cite:

```bibtex
@article{zheng2026open,
  title={Open-Ended Video Game Glitch Detection with Agentic Reasoning and Temporal Grounding},
  author={Zheng, Muyang and Zhou, Tong and Wu, Geyang and Lin, Zihao and Wang, Haibo and Huang, Lifu},
  journal={arXiv preprint arXiv:2604.07818},
  year={2026}
}
```
