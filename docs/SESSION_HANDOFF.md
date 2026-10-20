# Session handoff (checkpoint 2026-10-20)

Read this first, then `docs/COLAB_HANDOFF.md` (full detail on Colab work) and
`docs/IMPLEMENTATION_STATUS.md` (status by area). Facts marked **verified** were run or
checked in this session; everything else is labelled as an estimate or a plan.

## A. Project overview

- **Project:** Multimodal Visual RAG. Retrieval and grounded answer generation over document
  page images (ViDoRe v3 computer science: 1,360 pages, 215 queries), comparing text, visual
  and hybrid retrieval and vision-language generators under controlled evidence conditions.
- **Repository:** https://github.com/AdebanjiAdelowo/multimodal-visual-rag (private), branch
  `main`, local path `~/Documents/GitHub/multimodal-visual-rag`.
- **Current objective:** run Qwen3-VL-2B generation on a free Colab T4, directly comparable
  with the completed SmolVLM-500M study. After that come the other pending GPU workloads
  (see `COLAB_HANDOFF.md`, "Feasibility, revised 2026-10-19").

## B. Completed work (sessions of 2026-10-17 to 2026-10-20)

| Commit | What | Verified by |
|---|---|---|
| `bcde76c`, `63eaadd` | CI fixed: install into the setup-uv venv; `actions/checkout@v7`, `astral-sh/setup-uv@v10.3.0` with `activate-environment: true` | CI run 39056801094 green |
| `63c6901` | Colab status, Hub-measured model sizes, revised feasibility table | docs only |
| `4e27669` | `torchvision>=0.19` added to the `models` extra (Qwen3-VL processor needs it; previously only pulled in via `colpali-engine`). Lock changed no versions | local models-only simulation; Colab attempt 2 |
| `2917646` | Colab smoke test results recorded | checksums verified locally |
| `d40a96f` | Qwen3-VL-2B experiment prepared: pinned configs, `src/mmrag/colab_generation.py`, `notebooks/colab_qwen3vl_2b_generation.ipynb`, `tests/test_colab_generation.py` | 58 tests locally and in CI |
| `2847873` | Source bundle checksum for the experiment | CI run 39492302157 green (58 tests on 3.11 and 3.12, Docker job) |

**Measured results (verified):** Colab smoke test attempt 2 **passed** (2026-10-20, source
`4e27669`): Tesla T4, 14.56 GiB total VRAM; Qwen3-VL-2B `@89644892` float16 loaded in 65.0 s;
synthetic answers `red` and `blue` were both correct; latency 3.22 s (first call) and 0.22 s;
peak 4.04 GiB allocated / 4.07 GiB reserved. Attempt 1 failed at processor load
(missing torchvision); its report is kept in Drive. No benchmark experiment has run on a GPU.

## C. Current state

- **Works:** everything local (retrieval, SmolVLM-500M generation results, API, CI, Docker in CI);
  the Colab smoke test; the experiment runner under the mock generator (13 tests).
- **Prepared, not executed:** Qwen3-VL-2B memory check and full run. The notebook gates both with
  `APPROVE_MEMORY_CHECK = False` / `APPROVE_FULL_RUN = False`.
- **Not uploaded:** the corpus archive and the source ZIP to Google Drive (owner action).
- **No processes running.** No Kaggle job submitted.
- **Constraints:** shared Mac (see `~/.claude/CLAUDE.md`). No GPU work on the Mac. Approval is
  required for pushes, GPU runs, uploads and downloads over 500 MB. Disk ranged from 3.9 to 50 GB
  free during the session; check `df -h` first. Known local quirks: macOS hides the venv `.pth`
  files (use `PYTHONPATH=src`); a broken IPv6 route (`MMRAG_FORCE_IPV4=1 HF_HUB_DISABLE_XET=1`).
- **Open gaps (documented):** the 8B NF4 run needs `--extra quant`, which the notebooks do not
  install; non-2B generation configs still lack pinned revisions.

## D. Decisions and reasoning

- **Evidence frozen, not re-retrieved.** The pages each SmolVLM system saw were identical
  across its text, image and image + text runs (verified, all 150 keys), so they are frozen into
  `smolvlm_500m_evidence.jsonl` and given to Qwen unchanged. Re-running retrieval on CUDA could
  reorder ties. Rejected: shipping the indexes and rerunning `hybrid_text` on Colab.
- **Questions re-sampled and checked.** The CLI's `sample_queries` (60, seed 13) plus the
  30-question out-of-domain sample (seed 13) reproduce the SmolVLM IDs in order (verified). The
  runner refuses a corpus that does not.
- **Generation on Colab, scoring locally on CPU.** This matches how SmolVLM was scored
  (`NLIScorer` falls back to CPU without CUDA) and keeps GPU time minimal.
- **Corpus copied, not rebuilt.** A local OCR rebuild might not reproduce the fingerprint. The
  deterministic tar keeps inputs identical to the SmolVLM run.
- **Safety:** the output directory is bound to the experiment fingerprint and source commit;
  run files are append-only; completed files are checksummed; writing into `results/` is refused;
  `score` never overwrites.
- **Uncontrolled differences (documented in COLAB_HANDOFF):** MPS vs CUDA kernels;
  model-specific image tokenization and chat templates; the SmolVLM revision was not pinned at run
  time (only cached snapshot: `a7da5b9`); Python 3.12.2 vs 3.12.3. The oracle variant is out of scope.

## E. Reproducibility

- Environment: `uv` 0.11.16, `uv.lock`, Python 3.12, extras `models` (+ `app`, `dev` for tests).
  Local: `.venv`. Colab: `uv sync --frozen --python 3.12 --extra models`.
- Test: `HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" -q` (58 pass);
  `.venv/bin/ruff check src tests app kaggle && .venv/bin/ruff format --check src tests app kaggle`.
- Seeds: question sampling 13; decoding greedy; 160 new tokens; float16.
- Data: `vidore/vidore_v3_computer_science@d5cc758…`, CC-BY-4.0, fingerprint `ed29ca4d367565dd`.
- Models: `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`;
  NLI `cross-encoder/nli-deberta-v3-xsmall` (local cache).

### Artifacts outside Git (gitignored; local only, no other backup)

| Path | What | Status |
|---|---|---|
| `colab_outputs/corpus/mmrag_corpus-vidore_v3_computer_science-ed29ca4d367565dd.tar` (+ `.json`) | 404,029,440 B corpus bundle, SHA-256 `aedb0646e075d22514e526e89ad6cf49c88bbde56217934657130827f7cb9c9c` | reproducible with `package-corpus`; not uploaded |
| `colab_outputs/source/mmrag-d40a96ff….zip` | experiment source bundle, SHA-256 `11278e2a611742472dd1ae9f5642534f94ae8ecebd2b9f88622c90c19a30eb5c` | reproducible with `scripts/build_colab_bundle.py` at `d40a96f`; not uploaded |
| `colab_outputs/qwen3vl_2b/smolvlm_500m_evidence.jsonl` (+ `.meta.json`) | frozen evidence, SHA-256 `59e776c0…f707280` | also inside the tar |
| `colab_outputs/smoke_test/20261020T092130-ef026d98/` | passing smoke-test report and examples | also in Drive |
| `data/processed/`, `artifacts/` | corpus, indexes, SmolVLM run files | pre-existing; never modify the SmolVLM runs |

## F. Next steps (prioritized)

1. **Owner:** upload the tar to `My Drive/mmrag/corpus/` and the `d40a96f` ZIP to
   `My Drive/mmrag/source/` (exact steps: COLAB_HANDOFF "Corpus transfer").
2. **Owner approval, then Colab:** in `notebooks/colab_qwen3vl_2b_generation.ipynb`, paste
   `SOURCE_COMMIT`/`SOURCE_SHA256` (COLAB_HANDOFF "Source bundle for this experiment") and
   `CORPUS_ARCHIVE`/`CORPUS_SHA256` into cells 3 and 5. Run cells 1–6, set
   `APPROVE_MEMORY_CHECK = True`, and run cell 8. **Acceptance:** report `status: passed`, peak
   reserved below 80% of VRAM, checksum verified. Download `qwen3vl_2b_memory_check/`, then verify
   and record the measurements in COLAB_HANDOFF.
3. **Separate approval:** set `APPROVE_FULL_RUN = True` and run cell 10 (540 generations, resumable).
   **Acceptance:** four `*.complete.json` files; `verify_checksums` passes.
4. **Local:** download to `colab_outputs/qwen3vl_2b_generation/` and run `score` (command in
   COLAB_HANDOFF "After the run"). Regenerate `results/SUMMARY.md` with `mmrag report`.
   Document the results against SmolVLM.
5. Later: CLIP retrieval (fits on the Mac), ColQwen2, captions, 4B, 8B (add `--extra quant`),
   ablations. Pin revisions first.

## G. Resume instructions for a new session

1. `cd ~/Documents/GitHub/multimodal-visual-rag && git fetch && git status -sb && git log --oneline -5`;
   expect `main` clean and level with `origin/main`.
2. Run the tests (section E); expect 58 passed.
3. Do not redo: CI fixes, the smoke test, the evidence freeze, the corpus packaging, or the runner.
4. Ask the owner whether steps F1 and F2 have happened. If outputs exist in `~/Downloads` or
   `colab_outputs/`, verify their checksums before recording anything.
5. Never launch GPU runs, uploads or pushes without the owner's explicit approval.
