# Implementation status and handoff

Last updated: 2026-10-11 (session 3). Branch `main`, no remote, nothing pushed.

This file is the entry point for resuming work. GPU experiments run on Kaggle; their state is in [KAGGLE_HANDOFF.md](KAGGLE_HANDOFF.md), which is the first thing to read when resuming. Per-experiment status and commands are in [experiments.md](experiments.md); measured numbers are in [../results/SUMMARY.md](../results/SUMMARY.md), which is generated from the JSON files in `results/` and never edited by hand.

## Status by area

Statuses: COMPLETE AND VERIFIED, IMPLEMENTED BUT NOT VERIFIED, PARTIALLY IMPLEMENTED, NOT STARTED, BLOCKED, PENDING GPU.

| Area | Status | Evidence | Next action |
|---|---|---|---|
| Repository setup | COMPLETE AND VERIFIED | `pyproject.toml`, `uv.lock`, `.venv` (Python 3.12), 42 tests pass, ruff clean | none |
| Dataset ingestion | COMPLETE AND VERIFIED | ViDoRe v3 computer science: 1,360 pages, 215 English queries, 1,049 qrels (`data/processed/`, gitignored); PDF and image ingestion tested | none |
| OCR | COMPLETE AND VERIFIED | Tesseract text for all pages; OCR vs parser text compared | none |
| Text retrieval | COMPLETE AND VERIFIED | BM25, dense, text hybrid in `results/retrieval/` | none |
| Visual retrieval | COMPLETE AND VERIFIED locally; larger models PENDING GPU | SigLIP2 and ColSmol-256M results committed | CLIP and ColQwen2 on Kaggle |
| Hybrid retrieval | COMPLETE AND VERIFIED | text, SigLIP and ColSmol hybrids evaluated with paired tests | none |
| Reranking | COMPLETE AND VERIFIED | cross-encoder on three hybrids | none |
| Caption-based retrieval | PENDING GPU | code, configs and unit tests exist; no captions generated | Kaggle stage `captions` |
| Grounded generation | COMPLETE AND VERIFIED for SmolVLM-500M; larger models PENDING GPU | `results/generation/`; 240 answers reproduced exactly in a repeat run | Kaggle stages `generation`, `generation_8b` |
| Citation verification | IMPLEMENTED; not informative for SmolVLM-500M | parser and metrics unit-tested; the model cites a page in at most 10% of answers | re-assess with Qwen3-VL results |
| Retrieval evaluation | COMPLETE AND VERIFIED | metrics, strata (modality, query origin), bootstrap, paired tests | none |
| Generation evaluation | COMPLETE AND VERIFIED | three-condition harness run end to end; paired tests between evidence modes in the report | none |
| Ablation experiments | PARTIALLY COMPLETE | retrieval ablations done; generation ablations (pages, resolution, text budget, model size) PENDING GPU | Kaggle stage `ablations` |
| FastAPI | COMPLETE AND VERIFIED | `tests/test_api.py`, manual endpoint check in session 2 | none |
| Streamlit | IMPLEMENTED BUT NOT VERIFIED | loaded headless with AppTest in session 2; no test in the suite | optional AppTest-based test |
| Docker | IMPLEMENTED BUT NOT VERIFIED | image never built | `docker build -t mmrag .`, then `curl localhost:8000/health` |
| CI | IMPLEMENTED BUT NOT VERIFIED | workflow never executed (no remote) | runs on first push, which needs the owner's approval |
| Kaggle runner | IMPLEMENTED; dry-run tested; first submission recorded in KAGGLE_HANDOFF.md | `tests/test_kaggle_runner.py` runs the whole control flow with stubbed commands | import results when the job finishes |
| README/documentation | COMPLETE for all local results | README, architecture, datasets, evaluation, research audit, experiments | add GPU findings after import |

## Session history

| Session | Ended | What happened |
|---|---|---|
| 1 | 2026-10-09 | Research, scaffold, corpus, OCR, indexes, first retrieval runs. Hit the usage limit. |
| 2 | 2026-10-10 morning | Generation layer, evaluation harness, API, UI, Docker, CI, Kaggle runner, docs, tests. All work committed (`2b9616d` to `a59df54`). Stopped at the usage limit with a clean working tree and every model experiment except text retrieval still pending, because the machine was under heavy load. |
| 3 | 2026-10-10 to 2026-10-11 | Audit and environment repair; resumable late-interaction index (`add3358`); SigLIP and ColSmol results (`a18db67`, `41a4640`); MPS memory fix (`f66d4fc`); report additions (`2b8fd03`); SmolVLM-500M generation results (`434515d`); Kaggle runner (`1d251d3`, `564cd25`) and first Kaggle submission. |

## Architectural decisions worth knowing

- The page is the retrieval unit; dense text indexes chunk internally and max-pool to pages.
- Exact search over NumPy arrays; no vector database at 1,360 pages.
- Every index directory is keyed by a hash of its parameters and the corpus fingerprint (`src/mmrag/index/store.py`).
- One VLM serves all four evidence modes so that evidence modality is the only variable.
- Result JSON files in `results/` are the only source for reported numbers; `artifacts/` (indexes, per-query runs) and `data/` are gitignored.
- The benchmark is used zero-shot. Nothing is trained or tuned on it, so there is no train/dev/test split; the fusion weights (1.0 and 0.25) were fixed in the configs before the runs and both are reported.

## Known issues

- **iCloud eviction.** `~/Documents` is iCloud-synced with Optimize Mac Storage on. When the disk is nearly full (13 to 19 GB free during session 3) macOS evicts file contents inside this repository, including `.git` objects, source files and `.venv` files. Symptoms: `git status` hangs for minutes, pytest fails with `ModuleNotFoundError: No module named 'mmrag'`. See "Resume commands" for the repair.
- **Hidden `.pth` files.** macOS sets the hidden flag on `.venv/lib/python3.12/site-packages/*.pth`, which makes Python skip the editable install. `chflags nohidden` fixes it only until the flag is set again; prefixing commands with `PYTHONPATH=src` always works.
- **Broken IPv6 route.** Python downloads hang; set `MMRAG_FORCE_IPV4=1 HF_HUB_DISABLE_XET=1`.
- **Latency numbers are single-run.** Local retrieval timings come from one process at a load average of about 6.
- **`pad_token_id` warning from transformers (harmless).** Loading SmolVLM or ColSmol prints `Model config: pad_token_id must be None or an integer within the vocabulary (between 0 and 31999), got 128002`. Traced on 2026-10-11: it is raised inside `PretrainedConfig.to_diff_dict`, which builds a default `SmolVLMConfig` (class default `pad_token_id=128002`, inherited from Idefics3, and a default 32,000-token text config) only to print a config diff. The loaded model is not affected: `text_config.pad_token_id` is 2, `text_config.vocab_size` is 49,280, the generation config uses pad 2 and end-of-sequence 49,279, the tokenizer agrees, and the model code reads the padding index from `text_config`. Generation runs with batch size 1, so no padding is applied in any case. No configuration change is needed.
- **MPS allocator cache growth (fixed).** Before the fix in `TransformersVLM.generate`, the MPS buffer cache grew with every distinct prompt length and reached 15.4 GB after about 170 prompts with SmolVLM-500M. The cache is now released after each generation; the same run holds at 3.6 GB.
- Model revisions are not pinned locally. The Kaggle runner records the commit of every model it used.

## Resume commands

```bash
cd ~/Documents/GitHub/multimodal-visual-rag

# 1. Repair iCloud eviction and the hidden .pth flag if tools hang or imports fail
find .git src tests configs docs app kaggle results artifacts data .venv -type f -flags +dataless -print0 | xargs -0 -n 100 brctl download
chflags nohidden .venv/lib/python3.12/site-packages/*.pth

# 2. Verify
PYTHONPATH=src make test && make lint   # expect 42 passed, ruff clean
git status --short && git log --oneline -5

# 3. Check resources before any model run
uptime; sysctl -n vm.swapusage; df -h /System/Volumes/Data | tail -1

# 4. Follow docs/KAGGLE_HANDOFF.md: retrieve, verify and import the GPU results
```

All models needed for the pending local runs are already in the Hugging Face cache (bge-small, MiniLM reranker, SigLIP2, ColSmol-256M, SmolVLM-500M, Qwen3-VL-2B, nli-deberta-v3-xsmall), so the runs work with `HF_HUB_OFFLINE=1`.
