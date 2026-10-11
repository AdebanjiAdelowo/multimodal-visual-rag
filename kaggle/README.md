# GPU runs on Kaggle

`run_gpu.py` runs the experiments that need more memory or time than the local machine has: ColQwen2 and CLIP retrieval, full-corpus page captioning and caption retrieval, generation with Qwen3-VL 2B, 4B and 8B, and the generation ablations. It also repeats the local retrieval and SmolVLM-500M generation runs so that accuracy, latency and memory can be compared between the laptop and the GPU on identical examples.

## Setup

With the Kaggle command-line client (the route used for this project; it runs as a batch job that needs no open browser):

```bash
make kaggle-kernel                                   # builds kaggle/build/{dataset,kernel} from HEAD
kaggle datasets create -p kaggle/build/dataset       # first time (private by default)
kaggle datasets version -p kaggle/build/dataset -m "source at $(git rev-parse --short HEAD)"   # later updates
kaggle kernels push -p kaggle/build/kernel --accelerator NvidiaTeslaT4
kaggle kernels status adebanjiadelowo/multimodal-visual-rag-gpu
```

`make kaggle-kernel` refuses to run with uncommitted changes, so the uploaded source always corresponds to one commit. That commit is stored in the dataset (`COMMIT`) and copied into the session record of the results.

By hand instead: run `make kaggle-bundle`, upload `kaggle/mmrag-src.zip` as a private dataset, create a notebook with a GPU and internet enabled, attach the dataset, and paste `run_gpu.py` into a cell.

No credentials are required inside the job: the benchmark and all models are public and ungated (checked against the Hugging Face API on 2026-10-11; all are Apache-2.0 or MIT except CLIP ViT-B/32, whose model card declares no licence tag).

## Stages

| Stage | What it runs | Main downloads |
|---|---|---|
| `data` | benchmark download, Tesseract OCR, out-of-domain queries | 0.5 GB |
| `retrieval` | all sixteen first-stage, hybrid and reranked strategies on one machine | SigLIP2 1.4 GB, CLIP 0.6 GB, ColSmol 0.5 GB, ColQwen2 8.3 GB |
| `captions` | captions for 1,360 pages with SmolVLM-500M and Qwen3-VL-4B, then BM25 and dense retrieval over each caption set | 1.0 GB, 8.3 GB |
| `generation` | SmolVLM-500M, Qwen3-VL-2B and Qwen3-VL-4B, each in four evidence modes plus the oracle variant | 1.0 GB, 4.0 GB, 8.3 GB |
| `ablations` | page count (1, 2, 5), image resolution (512, 768, 1536) and text budget (800, 5000 characters) with Qwen3-VL-4B | none new |
| `generation_8b` | Qwen3-VL-8B in 4-bit, same design as `generation` | 16.3 GB |

`data` must run first in every new session. The other stages are independent of each other and run in the order listed, which is priority order: when the time budget is reached, the remaining commands are skipped and recorded rather than cut off mid-run.

## Options

| Variable | Default | Meaning |
|---|---|---|
| `MMRAG_STAGES` | all six | comma-separated stages to run |
| `MMRAG_GENERATORS` | `smolvlm_500m,qwen3vl_2b,qwen3vl_4b` | generators used by the `generation` stage |
| `MMRAG_N_QUERIES` | `60` | answerable questions per generator; `0` uses all 215 |
| `MMRAG_N_OOD` | `30` | out-of-domain questions |
| `MMRAG_TAG` | `t4` | suffix of the output directories |
| `MMRAG_BUDGET_HOURS` | `10.5` | no new command starts after this much wall-clock time |

The defaults reproduce the local protocol (60 answerable questions, 30 out-of-domain questions, seed 13, text-hybrid retrieval), so every generator on every machine answers exactly the same examples and can be compared pairwise. With the defaults each generator produces 600 answers.

A batch job takes its settings from the defaults above, since a script kernel has no cell in which to set variables. To run a subset, edit the defaults at the top of the copy in `kaggle/build/kernel/run_gpu.py` before pushing, and note the change in the handoff file.

## Output

`/kaggle/working/mmrag_results.zip` is rewritten after every command, so a job that stops early still leaves an archive of what finished. It contains:

- `results_t4/`: metric JSON files and `SUMMARY.md`;
- `artifacts/runs_t4/`: per-query records;
- `session_t4.json`: source commit, stage status and durations, failed and skipped commands, the protocol settings, the commit of every model used and the installed package versions;
- `SHA256SUMS`: a checksum for every file in the archive.

The source copy, the benchmark and the model weights live in `/kaggle/temp`, so the notebook output holds only the archive. It contains no page images and no page text. The per-query generation records do contain the benchmark's questions and reference answers, which are CC-BY-4.0.

Retrieve and verify:

```bash
kaggle kernels output adebanjiadelowo/multimodal-visual-rag-gpu -p /tmp/mmrag_kaggle
cd /tmp/mmrag_kaggle && unzip -q mmrag_results.zip -d unpacked && cd unpacked
shasum -a 256 -c SHA256SUMS                 # every line must say OK
python -m json.tool session_t4.json | head -40   # source_commit, stage status
```

Results are written under `results_t4/`, never `results/`, so copying them into the repository cannot overwrite results measured locally. After verification, copy `results_t4/` and `session_t4.json` into the repository root and `artifacts/runs_t4/` into `artifacts/`, then:

```bash
mmrag report --results results_t4 --runs artifacts/runs_t4 --out results_t4/SUMMARY.md
git add results_t4 session_t4.json && git commit -m "Add T4 results"
```

## Memory notes for a 16 GB T4

- Qwen3-VL-4B in float16 uses about 8.3 GB of weights plus activations; three 1024 px pages fit.
- Qwen3-VL-8B is loaded in 4-bit NF4 (about 6 GB).
- ColQwen2 in float16 uses about 4.5 GB; the page index for 1,360 pages is stored in float16.
- One generator is held in memory at a time. A command that fails (for example, out of memory) is recorded in `session_t4.json` and the remaining commands still run.
- Package versions on Kaggle are not the ones pinned in `uv.lock`; the versions actually used are recorded in `session_t4.json`.
