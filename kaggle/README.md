# GPU runs on Kaggle

`run_gpu.py` runs the experiments that need more memory than the local machine has: ColQwen2 retrieval, full-corpus captioning, Qwen3-VL 4B and 8B generation, and the generation ablations.

## Setup

1. From the repository root, create a source archive of the current commit:

   ```bash
   make kaggle-bundle        # writes kaggle/mmrag-src.zip
   ```

2. Upload `kaggle/mmrag-src.zip` to Kaggle as a private dataset.
3. Create a notebook with the accelerator set to "GPU T4 x2" and internet enabled. Attach the dataset.
4. Paste the contents of `run_gpu.py` into a cell. Edit `STAGES` if a subset is wanted. Run.

No credentials are required: the benchmark and all models are public and ungated.

## Stages

| Stage | What it runs | Main downloads |
|---|---|---|
| `data` | benchmark download, Tesseract OCR, out-of-domain queries | 0.5 GB |
| `retrieval` | all sixteen first-stage, hybrid and reranked strategies on one machine | SigLIP2 1.5 GB, CLIP 0.6 GB, ColSmol 0.5 GB, ColQwen2 8.8 GB |
| `captions` | captions for 1,360 pages with SmolVLM-500M and Qwen3-VL-4B, then caption retrieval | 1.0 GB, 8.9 GB |
| `generation` | four generators, four evidence modes, oracle variant, all 215 queries, 60 out-of-domain | 4.3 GB, 8.9 GB, 17.5 GB |
| `ablations` | page count, image resolution, text budget with Qwen3-VL-4B on 100 queries | none new |

All stages together may exceed one 12-hour session. `generation` is the longest; run it separately if needed. Generation runs append to JSONL files and resume where they stopped if the working directory is kept.

## Output

`/kaggle/working/mmrag_results.zip` contains `results/` (metric JSON and `SUMMARY.md`), `artifacts/runs/` (per-query records) and `session.json` (stage status and duration). Unpack it over the repository root, run `mmrag report`, and commit `results/`.

The archive contains no page images and no page text. The per-query generation records do contain the benchmark's questions and reference answers, which are CC-BY-4.0.

## Memory notes for a 16 GB T4

- Qwen3-VL-4B in float16 uses about 9 GB of weights plus activations; three 1024 px pages fit.
- Qwen3-VL-8B is loaded in 4-bit NF4 (about 6 GB).
- ColQwen2 in float16 uses about 4.5 GB; the page index for 1,360 pages is stored in float16.
- One generator is held in memory at a time.
