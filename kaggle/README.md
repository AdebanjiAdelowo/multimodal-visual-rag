# GPU runs on Kaggle

`run_gpu.py` runs the experiments that need more memory or time than the local machine has: ColQwen2 and CLIP retrieval, full-corpus page captioning and caption retrieval, generation with Qwen3-VL 2B, 4B and 8B, and the generation ablations. It also repeats the local retrieval and SmolVLM-500M generation runs so that accuracy, latency and memory can be compared between the laptop and the GPU on identical examples.

## Setup

1. From the repository root, create a source archive of the current commit:

   ```bash
   make kaggle-bundle        # writes kaggle/mmrag-src.zip (about 0.3 MB, source only)
   ```

2. Upload `kaggle/mmrag-src.zip` to Kaggle as a private dataset.
3. Create a notebook with the accelerator set to "GPU T4 x2" and internet enabled. Attach the dataset.
4. Paste the contents of `run_gpu.py` into a cell. To run a subset, set the environment variables below in a cell above it. Run.

No credentials are required: the benchmark and all models are public and ungated (checked against the Hugging Face API on 2026-10-11; all are Apache-2.0 or MIT except CLIP ViT-B/32, whose model card declares no licence tag).

## Stages

| Stage | What it runs | Main downloads |
|---|---|---|
| `data` | benchmark download, Tesseract OCR, out-of-domain queries | 0.5 GB |
| `retrieval` | all sixteen first-stage, hybrid and reranked strategies on one machine | SigLIP2 1.4 GB, CLIP 0.6 GB, ColSmol 0.5 GB, ColQwen2 8.3 GB |
| `captions` | captions for 1,360 pages with SmolVLM-500M and Qwen3-VL-4B, then BM25 and dense retrieval over each caption set | 1.0 GB, 8.3 GB |
| `generation` | each generator in four evidence modes plus the oracle variant | 1.0 GB, 4.0 GB, 8.3 GB, 16.3 GB |
| `ablations` | page count (1, 2, 5), image resolution (512, 768, 1536) and text budget (800, 5000 characters) with Qwen3-VL-4B | none new |

`data` must run first in every new session. The other stages are independent of each other.

## Options

| Variable | Default | Meaning |
|---|---|---|
| `MMRAG_STAGES` | all five | comma-separated stages to run |
| `MMRAG_GENERATORS` | `smolvlm_500m,qwen3vl_2b,qwen3vl_4b,qwen3vl_8b_4bit` | generators used by the `generation` stage |
| `MMRAG_N_QUERIES` | `60` | answerable questions per generator; `0` uses all 215 |
| `MMRAG_N_OOD` | `30` | out-of-domain questions |
| `MMRAG_TAG` | `t4` | suffix of the output directories |

The defaults reproduce the local protocol (60 answerable questions, 30 out-of-domain questions, seed 13, text-hybrid retrieval), so every generator on every machine answers exactly the same examples and can be compared pairwise. With the defaults each generator produces 600 answers.

Suggested split into sessions, each well inside the 12-hour limit:

```python
import os

# Session A: retrieval and captions
os.environ["MMRAG_STAGES"] = "data,retrieval,captions"
# Session B: small generators
os.environ["MMRAG_STAGES"] = "data,generation"
os.environ["MMRAG_GENERATORS"] = "smolvlm_500m,qwen3vl_2b"
# Session C: large generators
os.environ["MMRAG_STAGES"] = "data,generation"
os.environ["MMRAG_GENERATORS"] = "qwen3vl_4b,qwen3vl_8b_4bit"
# Session D: ablations
os.environ["MMRAG_STAGES"] = "data,ablations"
```

Use a different `MMRAG_TAG` only if the hardware differs; sessions with the same tag produce archives that merge cleanly because each holds different files.

## Output

`/kaggle/working/mmrag_results.zip` is rewritten after every stage, so a session that times out still leaves an archive of the stages that finished. It contains:

- `results_t4/`: metric JSON files and `SUMMARY.md`;
- `artifacts/runs_t4/`: per-query records;
- `session_t4.json`: stage status and duration, failed commands, the protocol settings, the commit of every model used and the installed package versions.

Results are written under `results_t4/`, never `results/`, so unpacking the archive over the repository root cannot overwrite results measured locally. After unpacking:

```bash
mmrag report --results results_t4 --runs artifacts/runs_t4 --out results_t4/SUMMARY.md
git add results_t4 && git commit -m "Add T4 results"
```

The archive contains no page images and no page text. The per-query generation records do contain the benchmark's questions and reference answers, which are CC-BY-4.0.

## Memory notes for a 16 GB T4

- Qwen3-VL-4B in float16 uses about 8.3 GB of weights plus activations; three 1024 px pages fit.
- Qwen3-VL-8B is loaded in 4-bit NF4 (about 6 GB).
- ColQwen2 in float16 uses about 4.5 GB; the page index for 1,360 pages is stored in float16.
- One generator is held in memory at a time. A command that fails (for example, out of memory) is recorded in `session_t4.json` and the remaining commands still run.
- Package versions on Kaggle are not the ones pinned in `uv.lock`; the versions actually used are recorded in `session_t4.json`.
