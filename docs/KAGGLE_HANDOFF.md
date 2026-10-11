# Kaggle handoff

Last updated: 2026-10-11 14:20 CEST.

## Current state: READY, NOT YET RUNNING

| Item | Value |
|---|---|
| Kaggle account | `adebanjiadelowo` |
| Source dataset | `adebanjiadelowo/multimodal-visual-rag-src` (private), version 1, status `ready`, uploaded 2026-10-11 14:15 CEST |
| Source commit in the dataset | `3a723f9bb9f90a4bd7b996ee5ce2349367708858` (file `COMMIT` in the dataset) |
| `mmrag-src.zip` SHA-256 | `6f79fbec3e3e593a76d7cce886492e4d105eedc600a5105fc7a8ff53b9f1e818` |
| Kernel slug | `adebanjiadelowo/multimodal-visual-rag-gpu` (private script kernel, GPU T4, internet on) |
| Kernel script | `kaggle/run_gpu.py` at commit `3a723f9`, SHA-256 `8069e919eaa44b0f376ed1d619ad9cac47bcb152bf2733dce2d426eb81d6091c` |
| Kernel version | none: the push on 2026-10-11 14:17 CEST was refused with `Maximum batch GPU session count of 2 reached` |

The two GPU batch slots of the account were held by other projects' jobs (`legal-research-rag-gpu` running, `research-innovation-monitor-gpu` queued). Nothing was created on Kaggle for the kernel, and nothing has run. No results exist yet.

## Next action: submit once a GPU slot is free

```bash
cd ~/Documents/GitHub/multimodal-visual-rag
make kaggle-submit KAGGLE_PYTHON=/opt/anaconda3/bin/python
make kaggle-status KAGGLE_PYTHON=/opt/anaconda3/bin/python     # expect QUEUED or RUNNING
```

`kaggle/build/kernel/` already holds the script and metadata for commit `3a723f9`. If that folder is missing, rebuild it without touching the uploaded dataset:

```bash
mkdir -p kaggle/build/kernel
git show 3a723f9:kaggle/run_gpu.py > kaggle/build/kernel/run_gpu.py
git show 3a723f9:kaggle/kernel-metadata.json > kaggle/build/kernel/kernel-metadata.json
shasum -a 256 kaggle/build/kernel/run_gpu.py     # must equal the hash in the table above
```

After a successful push, record the kernel version number and the submission time in the table above, then stop. Do not wait for the job.

If the source changes before submission, run `make kaggle-kernel`, upload a new dataset version (`python kaggle/kaggle_ipv4.py datasets version -p kaggle/build/dataset -m "source at <sha>"`), and update the commit and hashes here.

Use `kaggle/kaggle_ipv4.py` (through the make targets) rather than the plain `kaggle` command: on this network the plain client hangs during uploads because of a broken IPv6 route.

## Experiment configuration

One batch job runs these stages in priority order, with the defaults in `kaggle/run_gpu.py`:

| Stage | Content | Estimated time on a T4 |
|---|---|---|
| `data` | download ViDoRe v3 computer science, Tesseract OCR, out-of-domain queries | 0.3 h |
| `retrieval` | 16 strategies, including CLIP ViT-B/32 and ColQwen2 (new) and the 14 already measured locally | 0.7 h |
| `captions` | captions for 1,360 pages with SmolVLM-500M and Qwen3-VL-4B; BM25 and dense retrieval over each | 2 to 3 h |
| `generation` | SmolVLM-500M, Qwen3-VL-2B, Qwen3-VL-4B; closed-book, text, image, image+text, oracle | 3 to 4 h |
| `ablations` | Qwen3-VL-4B: pages 1/2/5, image side 512/768/1536, text budget 800/5000 | 2 h |
| `generation_8b` | Qwen3-VL-8B in 4-bit, same design | 2 to 3 h |

Protocol: 60 answerable questions (seed 13, the same sample as the local run), the same 60 with relevant pages withheld, 30 out-of-domain questions, text-hybrid retrieval, three pages of evidence, greedy decoding. Time estimates are rough and untested. The job stops starting new commands after 10.5 hours and records the ones it skipped, so the later stages may be incomplete; they can be run in a second job by setting `STAGES` in the kernel copy of the script.

## Expected outputs

Notebook output `mmrag_results.zip` containing:

- `results_t4/retrieval/vidore_v3_computer_science/*.json` (up to 20 strategies) and `results_t4/generation/vidore_v3_computer_science/*.json` (5 files per generator, plus 8 ablation files);
- `results_t4/SUMMARY.md`;
- `artifacts/runs_t4/` per-query records;
- `session_t4.json` (source commit, stage status, failed and skipped commands, model commits, package versions);
- `SHA256SUMS`.

## When the job has finished: retrieve and validate before importing

```bash
cd ~/Documents/GitHub/multimodal-visual-rag
OUT=$(mktemp -d)
/opt/anaconda3/bin/python kaggle/kaggle_ipv4.py kernels output adebanjiadelowo/multimodal-visual-rag-gpu -p $OUT
cd $OUT && unzip -q mmrag_results.zip -d unpacked && cd unpacked
shasum -a 256 -c SHA256SUMS | grep -v ': OK$'          # must print nothing
python3 -c "import json; s=json.load(open('session_t4.json')); print(s['source_commit'], s['elapsed_hours']); [print(e['stage'], e['status'], e['failed_commands'], e['skipped_commands']) for e in s['stages']]"
```

Checks before copying anything into the repository:

1. `source_commit` equals `3a723f9bb9f90a4bd7b996ee5ce2349367708858` (or the commit recorded above if it was changed).
2. Every checksum is OK.
3. Provenance: the 14 retrieval strategies already measured locally must reproduce the local quality metrics closely (locally, repeated runs gave identical rankings; on a different GPU and library version small differences are possible and should be reported, not hidden). `nDCG@10` for `hybrid_text` is 0.689 locally.
4. SmolVLM-500M on the T4 answers the same 60 questions as the local run; compare token F1 (local: closed-book 0.251, text 0.343, image 0.299, image+text 0.290, oracle 0.306). Differences are a finding for RQ6, not an error to fix.
5. Each generation result file has `conditions.answerable.n == 60`, `missing_evidence.n == 60` (except closed-book) and `out_of_domain.n == 30`.
6. Read the kernel log for `COMMAND FAILED` and `SKIPPED` lines and list them in `docs/experiments.md`.

Then import:

```bash
cp -R results_t4 session_t4.json ~/Documents/GitHub/multimodal-visual-rag/
mkdir -p ~/Documents/GitHub/multimodal-visual-rag/artifacts && cp -R artifacts/runs_t4 ~/Documents/GitHub/multimodal-visual-rag/artifacts/
cd ~/Documents/GitHub/multimodal-visual-rag
PYTHONPATH=src .venv/bin/mmrag report --results results_t4 --runs artifacts/runs_t4 --out results_t4/SUMMARY.md
PYTHONPATH=src make test && make lint
```

Local results in `results/` are never overwritten by this procedure. After import: add the GPU findings (captions for RQ2, ColQwen2 for RQ1, model size and abstention for RQ5, laptop against T4 for RQ6) to `README.md`, `docs/research_audit.md` and `docs/experiments.md`, with negative results kept, and commit.

## Rules

- Inspect existing results before any rerun. If the kernel shows `COMPLETE`, retrieve its output; do not submit again.
- Do not poll or wait for the job inside a Claude session.
- Do not push to GitHub, and do not cancel or modify other projects' Kaggle jobs.
