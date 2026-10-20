# Colab GPU smoke-test handoff

Prepared 2026-10-12. No Colab GPU measurements yet. The initial test uses only
`Qwen/Qwen3-VL-2B-Instruct`, pinned to `89644892e4d85e24eaac8bacfd4f463576704203`,
float16, no quantization, two generated 512-pixel square images, 48 output tokens.
It exercises the production `TransformersVLM` backend, not a separate pipeline.
A pass requires model loading, nonempty inference, persistent artifact read-back
and checksums. It does not establish correctness or benchmark performance.

## Run manually

1. Run `python3 scripts/build_colab_bundle.py` from this repository after committing.
   This archives HEAD and writes its commit and SHA-256 to
   `colab_outputs/source/manifest.json`. Do not bundle working-tree files or credentials.
2. Open https://colab.research.google.com, select **Upload notebook**, and upload
   `notebooks/colab_gpu_smoke_test.ipynb`.
3. Select **Runtime → Change runtime type → T4 GPU**. Run the first cell; stop
   if `nvidia-smi` fails. No paid runtime purchase is authorized.
4. Run the Drive cell and approve access yourself. Output goes to
   `MyDrive/mmrag/colab_outputs/smoke_test/`. If Drive authorization is unavailable,
   stop: `/content` alone will not preserve outputs after a disconnect.
5. Paste the two values from the local manifest into the source cell. Run it and
   upload only the source ZIP. Its checksum is checked before extraction.
6. Run installation and inference cells. Installation uses uv 0.11.16 and
   `uv sync --frozen --python 3.12 --extra models` in an isolated environment.
   The first model download is about 4.3 GB; do not download 4B or 8B.
7. Return the newest report directory, `source_provenance.json`, and `console.log`.
   Keep failure reports too. Do not start the experimental suite.

The notebook is unexecuted in Git. Report filenames are unique; existing verified
results are never modified. If installation fails before the runner starts,
retain the cell output: that is an environment failure, not an inference failure.

## Stage feasibility (estimates, not Colab measurements)

Memory includes approximate weights and a conservative activation allowance;
image count/resolution and sequence length can make peaks substantially higher.
Times below are planning ranges for the runner's actual 60-question protocol,
not extrapolations from a completed GPU test.

| Pending component | Download | Approximate VRAM | Time on T4 | Resume / independent execution |
|---|---|---|---|---|
| CLIP ViT-B/32 retrieval | 0.6 GB | 1–2 GB | 5–15 min | complete index cache; select its config independently |
| ColQwen2 retrieval | 8.8 GB base + adapter | 6–12 GB | 20–60 min | page-batch index resume; independent config; T4 plausible |
| Caption retrieval | 1 GB SmolVLM + 8.9 GB Qwen4B | 2–4 / 10–15 GB | 2–4 h | captions resume per page; retrieval follows captions |
| Qwen3-VL-2B generation | 4.3 GB | 6–10 GB | 0.5–2 h | JSONL resume; independent CLI invocation; T4 plausible |
| Qwen3-VL-4B generation | 8.9 GB | 11–16+ GB | 1–3 h | JSONL resume; T4 uncertain with three images; larger GPU preferable |
| Qwen3-VL-8B NF4 | 17.5 GB before quantization | 7–14+ GB | 2–4 h | separate `generation_8b` stage; CUDA bitsandbytes; unverified |
| Generation ablations (4B) | same 8.9 GB | 12–20+ GB | 2–4 h | separate commands and JSONL runs; five pages / 1536 px may exceed T4 |

The runner's `retrieval` stage includes fourteen completed local strategies too;
use individual configs to avoid rerunning them. `generation` groups SmolVLM,
Qwen2B and Qwen4B; use a Qwen-specific CLI invocation for independent execution.
`generation_8b` is separate. All benchmark experiments depend on materialized
corpus and OCR; generation also uses text-hybrid retrieval and NLI scoring.
The smoke test needs none of those benchmark assets or indexes.

Resume only survives Colab replacement if corpus, indexes and per-query run files
are stored persistently; the existing Kaggle runner's scratch defaults do not
provide that on Colab. Do not invoke it unchanged from this notebook.

## Browser execution attempt, 2026-10-12

Notebook uploaded successfully to:
https://colab.research.google.com/drive/1R6Bq_A5vZVbA9jkZThca0Q7xVdLfGdYi

T4 was selected and only the first diagnostic cell was queued. The browser
remained at **Connecting / Allocating runtime**, with the cell unexecuted when
handed off. This is not a confirmed quota failure; the underlying reason is
unknown. No model download, inference, or Drive mount was started. No GPU type,
latency or peak VRAM was measured. If allocation completes, continue from the
Drive cell using the instructions above. If it does not, retry allocation manually.

Prepared bundle (297 KB):
- commit `6968e5fa3f6cee4b93f49a5eaf5d07e3607fd707`
- SHA-256 `4b208806a5b75e54b7d8029b771170857c3646dad16157374cc27d309c969402`
- `colab_outputs/source/mmrag-6968e5fa3f6cee4b93f49a5eaf5d07e3607fd707.zip`

Local validation: 45 tests passed; Ruff lint and format checks passed; notebook
code parsed with Python AST; archived source checksum and exclusions verified;
no model weights or credential files tracked. Existing `results/` unchanged.
An additional mocked no-CUDA check verified failure reporting without model loading.

## Status check, 2026-10-19: BLOCKED (no runtime reachable)

Smoke test **not run**. The agent session had no browser signed in to Google,
and the Drive mount needs the owner's own authorization, so no Colab runtime was
reachable. No GPU type, VRAM, latency, load time or peak memory has been measured.
No Colab outputs exist: `colab_outputs/` holds only the source bundle.

Re-verified locally on 2026-10-19:
- Bundle SHA-256 still matches `manifest.json` (`4b208806…c969402`, commit `6968e5f`).
  Commits since `6968e5f` change only docs and `.github/workflows/ci.yml`, so the
  bundled code equals HEAD and the bundle does not need rebuilding.
- `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` exists on
  the Hub and is the current `main` tip; weights 4.26 GB (Hub metadata), Apache-2.0.

### Values to paste into the source cell

```python
SOURCE_COMMIT = "6968e5fa3f6cee4b93f49a5eaf5d07e3607fd707"
EXPECTED_SHA256 = "4b208806a5b75e54b7d8029b771170857c3646dad16157374cc27d309c969402"
```

Upload `colab_outputs/source/mmrag-6968e5fa3f6cee4b93f49a5eaf5d07e3607fd707.zip`.

### Manual run (owner)

1. Open https://colab.research.google.com/drive/1R6Bq_A5vZVbA9jkZThca0Q7xVdLfGdYi
   and set **Runtime → Change runtime type → T4 GPU** (free tier; no purchase).
2. Run cells 1–5 in order. Approve Drive access in cell 2 yourself. Paste the two
   values above into cell 3. The free-VRAM guard needs 8 GiB, which a T4 has.
3. Cell 5 passes only if `exit status: 0` and `report['status'] == 'passed'`.
4. Download `MyDrive/mmrag/colab_outputs/smoke_test/` (newest run directory,
   `source_provenance.json`, `console.log`) to `colab_outputs/smoke_test/` locally.
   Keep failure reports as they are. Do not run anything beyond cell 5.

The report records the GPU name, total/free VRAM, CUDA, torch and transformers
versions, load latency, per-example latency and peak allocated/reserved memory.

### Feasibility, revised 2026-10-19

Downloads are **measured from Hub metadata** (safetensors totals, current `main`).
VRAM and runtime are **estimates** carried over from the table above. None of them
are Colab measurements. Resume support was checked in code, not run on Colab.

| Workload | Model (Hub `main` sha) | Download | VRAM (est.) | T4 time (est.) | Resume | Independent | Free T4 (15 GB) |
|---|---|---|---|---|---|---|---|
| CLIP retrieval | `openai/clip-vit-base-patch32` (`3d74acf`) | ~0.6 GB PyTorch weights; repo has no safetensors (1.8 GB incl. TF/Flax) | 1–2 GB | 5–15 min | full-index cache | yes, `visual_clip.yaml` | yes; also feasible on the Mac, does not need a GPU slot |
| ColQwen2 retrieval | `vidore/colqwen2-base` (`9fe8a71`) + `vidore/colqwen2-v1.0` adapter (`2b6ac8f`) | 8.84 + 0.07 GB | 6–12 GB | 20–60 min | per-batch index resume (`index/late.py`) | yes, `late_colqwen2.yaml` | plausible |
| Caption retrieval | SmolVLM-500M (`a7da5b9`), then Qwen3-VL-4B (`ebb281e`) | 1.02 GB; 8.88 GB | 2–4 / 10–15 GB | 2–4 h | per-page caption JSONL | captions first, then retrieval | yes for SmolVLM; 4B tight |
| Qwen3-VL-2B generation | `89644892` (pinned) | 4.26 GB | 6–10 GB | 0.5–2 h | per-key JSONL (`generation_eval.run_system`) | yes, per-model CLI | plausible; smoke test measures it |
| Qwen3-VL-4B generation | `ebb281e` | 8.88 GB | 11–16+ GB | 1–3 h | per-key JSONL | yes | uncertain with three 1024-px pages |
| Qwen3-VL-8B NF4 | `0c351dd` | 17.53 GB before quantization | 7–14+ GB | 2–4 h | per-key JSONL | yes, `generation_8b` | uncertain; also needs `--extra quant` (not installed by the notebook) and ~18 GB of free disk on the runtime |
| Generation ablations (4B) | `ebb281e` | 8.88 GB (shared) | 12–20+ GB | 2–4 h | per-run JSONL | yes, separate commands | five pages / 1536 px likely exceeds a T4 |

Gaps to close before the first benchmark experiment, found 2026-10-19:
- Generation configs do not pin `vlm.revision`; only the smoke test does. Pin it
  in any Colab experiment config so the model revision is reproducible.
- All benchmark workloads need the materialized corpus with OCR text on persistent
  storage. Either rebuild it on Colab (`mmrag fetch-vidore`, 512 MB, then Tesseract
  OCR, which may not reproduce the local fingerprint), or copy the existing
  `data/processed/vidore_v3_computer_science` (389 MB, CC-BY-4.0) to Drive and
  verify its corpus fingerprint. Copying is preferred: the paired comparison with
  the completed SmolVLM-500M run then uses identical inputs.

### Recommended first GPU experiment (not prepared; gated on a passing smoke test)

Qwen3-VL-2B generation over the same 60 questions × 4 evidence conditions
(closed book, text, image, image + text) as the completed SmolVLM-500M run,
with the model pinned to `89644892`. It is the cheapest workload that answers a
pending research question: it directly measures the effect of model scale against
an existing result and checks whether citation verification becomes informative.
The model is the same one the smoke test loads. CLIP is cheaper, but it is a weak
baseline that fits on the Mac and does not need Colab. The runner, persistent
output layout, checkpointing and checksums will be prepared only after the smoke
test passes and its measured VRAM is known.

## Attempt 1, 2026-10-20: FAILED at model loading (missing torchvision)

Run directory: `MyDrive/mmrag/colab_outputs/smoke_test/20261020T091149-3eb680a4/`
(source `6968e5f`, bundle SHA-256 verified by the notebook). Report and checksum
read back successfully in cell 5; `status: failed`, `model_loading: failed`,
`inference: not_started`. Keep this directory; it is the evidence for this entry.

Measured by the runner (from `report.json`):

| Item | Value |
|---|---|
| GPU | Tesla T4, CUDA available |
| VRAM total / free at start | 15,637,086,208 B (14.56 GiB) / 15,527,116,800 B (14.46 GiB) |
| Python / CUDA (torch build) | 3.12.3 / 13.0 |
| torch / transformers / accelerate / huggingface-hub / pillow | 2.13.0 / 5.19.0 / 1.15.0 / 2.2.0 / 12.3.0 |

Not measured: model load time, inference latency, peak memory. The failure came in
`AutoProcessor.from_pretrained`, before any model weights were loaded.

Cause: `ImportError: Qwen3VLVideoProcessor requires the Torchvision library`.
`torchvision` was in `uv.lock` only as a dependency of `colpali-engine` (extra
`late`). The local `.venv` has that extra, so Qwen3-VL loaded locally. The notebook
installs only `--extra models`. Fix: `torchvision>=0.19` is now declared in the
`models` extra. The relock (uv 0.11.16) changes no package or version, adds
`torchvision 0.28.0` to `models`, and only rewrites environment markers.

Local check of the fix, run without extra disk (3.9 GB free): a site-packages view
containing exactly the `uv export --extra models` packages, run with `python -S`,
loads `Qwen3VLProcessor` / `Qwen3VLVideoProcessor` at the pinned revision from the
local cache. It reproduces the Colab error when torchvision is removed. Weights
were not loaded and no GPU was used.

Retry: build a new bundle from the commit that contains this fix
(`python3 scripts/build_colab_bundle.py`), then rerun cells 3–5 with the new commit
and checksum from `colab_outputs/source/manifest.json`. Cell 3 extracts into a
directory named after the commit, so the retry does not collide with attempt 1.

## Attempt 2, 2026-10-20: PASSED

Run directory: `MyDrive/mmrag/colab_outputs/smoke_test/20261020T092130-ef026d98/`,
copied to the local `colab_outputs/smoke_test/20261020T092130-ef026d98/` (gitignored).
Source `4e27669` (bundle SHA-256 `2a686904…9da0c5`, verified by the notebook).
Verified locally on 2026-10-20: `report.sha256` matches `report.json`; both example
files match the SHA-256 and payload recorded in the report; status, model loading
and inference are all `passed` with no errors. `source_provenance.json` and
`console.log` were not downloaded; they remain in Drive.

Measured on Colab (single run; not a benchmark):

| Item | Value |
|---|---|
| GPU / VRAM total / free at start | Tesla T4 / 14.56 GiB / 14.46 GiB |
| Python / CUDA / torch / transformers | 3.12.3 / 13.0 / 2.13.0 / 5.19.0 |
| Model | `Qwen/Qwen3-VL-2B-Instruct@89644892…`, float16, no quantization |
| Load time | 65.0 s (likely includes the first weight download; cold start) |
| Example 0 (red square) | answer `red`, 3.22 s, 275 input / 2 output tokens (first call, includes warm-up) |
| Example 1 (blue square) | answer `blue`, 0.22 s, 275 input / 2 output tokens |
| Peak allocated / reserved | 4.04 GiB / 4.07 GiB (4,340,698,624 / 4,370,464,768 B) |

What this establishes: the locked environment installs on Colab, CUDA works, the
pinned 2B model loads through the production `TransformersVLM` backend, and the
Drive output and checksum path works. What it does not establish: answer quality,
or memory and latency at benchmark settings. The 4.04 GiB peak covers 275 input
tokens at 512 px. Three 1024-px pages produce far more visual tokens, so the
6–10 GB estimate for 2B generation stays an estimate until measured.

## Qwen3-VL-2B generation experiment: prepared 2026-10-20, NOT RUN

Runner `src/mmrag/colab_generation.py`, notebook `notebooks/colab_qwen3vl_2b_generation.ipynb`,
tests `tests/test_colab_generation.py`. Neither the memory check nor the full run has been
executed. Both notebook stages stop unless the owner sets their approval flag to `True`.

### Design: what is held fixed against SmolVLM-500M

| Element | How it is controlled |
|---|---|
| Questions | Re-sampled on Colab with the CLI's `sample_queries` (60, seed 13) and the out-of-domain sample (30, seed 13), then required to equal the frozen key list in order. Verified locally: identical IDs, order, questions, reference answers and gold pages. |
| Conditions | answerable, missing_evidence (gold pages withheld), out_of_domain; closed book skips missing_evidence, as in `mmrag eval-generation`. 540 generations: 90 + 150 + 150 + 150. |
| Evidence | Not re-retrieved. The three pages each SmolVLM system saw (identical across its text, image and image + text runs for all 150 keys) are frozen into `smolvlm_500m_evidence.jsonl` (SHA-256 `59e776c0…f707280`) and passed unchanged. Retrieval scores are not used by evidence selection or the prompt. |
| Evidence settings | `max_pages` 3, `image_max_side` 1024, `ocr_tesseract` text, 2,500 characters per page. The runner refuses to start if any Qwen config differs from the SmolVLM config. |
| Generation | Greedy decoding (backend), 160 new tokens, float16; model pinned to `89644892`. |
| Scoring | `mmrag.colab_generation score`, run locally: the same `score_records` and `summarize` code, `nli-deberta-v3-xsmall` on CPU, `parser_markdown` for faithfulness, as for SmolVLM. |

Differences that a strictly controlled comparison cannot remove:
- **Hardware and kernels:** SmolVLM ran on Apple MPS, Qwen runs on a CUDA T4; both float16. Latency and memory are not comparable across the two.
- **Model-specific inputs:** the same 1024-px images and prompt text go in, but SmolVLM tiles images into 512-px crops while Qwen3-VL uses native-resolution patches. Chat templates also differ. Both are properties of the models being compared.
- **SmolVLM revision:** not pinned when it ran (2026-10-11). The only cached snapshot is `a7da5b9…`, which is still the Hub `main`, so it is very likely, but not provably, the revision used.
- **Software:** Python 3.12.3 on Colab vs 3.12.2 locally; torch 2.13.0 on both. The SmolVLM result files do not record the transformers version (local is now 5.19.0, as on Colab).
- **Out of scope:** the SmolVLM `image_rag__oracle` variant (gold pages) is not part of this four-system run.

### Output and recovery

`MyDrive/mmrag/colab_outputs/qwen3vl_2b_generation/`:
- `provenance.json`: written once. It holds the experiment description and its SHA-256 (configs, model revision, seed, corpus manifest, evidence). A later session with a different configuration or source commit is refused.
- `sessions.jsonl`: one start and one end line per session, with packages, GPU, hardware and status.
- `runs/qwen3vl_2b_<system>.jsonl`: one answer per line, appended and flushed as generated (`generation_eval.run_system`).
- `runs/*.calls.jsonl`: per-call latency, tokens and peak allocated/reserved VRAM.
- `runs/*.complete.json`: written when a system finishes; holds the run file's SHA-256 and the peak VRAM over all its calls. A run file that changes after completion is refused.
- `CHECKSUMS.sha256`: rewritten at the end of every session, including interrupted ones.

After a disconnect: reconnect, rerun cells 1–6, then rerun the interrupted stage. Completed
answers are skipped. A half-written last line is moved to `runs/<name>.partial-<time>` and
regenerated. Drive writes are asynchronous, so the last few answers before a hard disconnect
may be lost and regenerated; they are never duplicated, because keys are checked.
Writing into the repository's `results/` is refused; results are imported only by `score`.

### Corpus transfer (owner action; nothing has been uploaded)

Local source: `~/Documents/GitHub/multimodal-visual-rag/data/processed/vidore_v3_computer_science/`
(ViDoRe v3 computer science, revision `d5cc758…`, CC-BY-4.0; 1,360 pages).
Already packaged by `python -m mmrag.colab_generation package-corpus` (byte-reproducible):

| Item | Value |
|---|---|
| Archive | `colab_outputs/corpus/mmrag_corpus-vidore_v3_computer_science-ed29ca4d367565dd.tar` |
| Size | 404,029,440 bytes (385.3 MiB), uncompressed tar (JPEG pages do not compress) |
| SHA-256 | `aedb0646e075d22514e526e89ad6cf49c88bbde56217934657130827f7cb9c9c` |
| Contents | corpus (1,368 files), frozen evidence, `MANIFEST.json` with a SHA-256 per file |
| Manifest SHA-256 / corpus fingerprint | `9beaeff5…bef02693` / `ed29ca4d367565dd` |

1. Check locally: `shasum -a 256 colab_outputs/corpus/mmrag_corpus-vidore_v3_computer_science-ed29ca4d367565dd.tar`
2. In drive.google.com, create `My Drive/mmrag/corpus/` and upload the `.tar` there unchanged
   (do not unzip or convert it). Create `My Drive/mmrag/source/` and upload the source ZIP there.
3. Notebook cell 5 copies the tar to `/content`, checks the SHA-256, extracts it with
   `tarfile` `filter='data'` (rejects absolute paths, links and traversal), then runs
   `verify-corpus`, which checks every file, the file set and the corpus fingerprint.
   `run` and `memory-check` verify the corpus again before using it.

Verified locally: two builds gave the same SHA-256; a safe extraction verified all 1,369 files
and rebuilt the 540 examples.

### Memory validation procedure (prepared, not executed)

Notebook cell 8, gated by `APPROVE_MEMORY_CHECK`. It runs `memory-check`, which:
- uses the real configs and the actual prompt path (`RAGPipeline.answer` with frozen evidence);
- picks the five heaviest `image_text_rag` prompts (three pages, the most OCR text; every
  evidence page is 1236×1600, so 791×1024 after resizing), plus the heaviest `image_rag` and
  `text_rag` prompt;
- records per prompt the image sizes after resizing, evidence characters, input/output tokens,
  `max_new_tokens`, latency, and peak allocated/reserved VRAM (reset before each prompt);
- passes only if peak reserved VRAM is below 80% of the GPU total (11.6 GiB on a T4). The first
  prompt's latency includes CUDA warm-up;
- writes `qwen3vl_2b_memory_check/memory_check/<time>/report.json` and `.sha256`, and records
  the planned prompts first, so an out-of-memory failure still shows which prompt it hit.
  It never writes experiment run files.

### Estimated Colab requirements (estimates, not measurements)

| Resource | Estimate | Basis |
|---|---|---|
| Peak VRAM, full run | 5–8 GiB | 4.04 GiB measured for weights plus one 275-token prompt; the heaviest prompt is about 4–5k tokens (three 791×1024 pages plus up to 7,500 characters). Unverified until the memory check runs. |
| Generation time | 0.5–1.5 h for 540 prompts | assumes 2–6 s per prompt on a T4 (SmolVLM-500M averaged 1.8 s on MPS) |
| Per-session setup | 5–10 min | uv install, 4.3 GB model download (65 s load measured), 385 MiB corpus copy and verification |
| Colab disk | ~11 GB | venv, model cache, tar and extracted corpus |
| Drive | ~0.4 GB | tar, source ZIP, outputs of a few MB |

The run fits one free-tier session if the estimates hold. If not, it resumes across sessions.

### After the run (local)

Download `qwen3vl_2b_generation/` to `colab_outputs/qwen3vl_2b_generation/`, then:

```bash
PYTHONPATH=src .venv/bin/python -m mmrag.colab_generation score \
  --output colab_outputs/qwen3vl_2b_generation \
  --corpus data/processed/vidore_v3_computer_science
```

This checks the run checksums and the corpus fingerprint, scores on CPU, and writes
`results/generation/vidore_v3_computer_science/qwen3vl_2b_*.json` plus scored JSONL under
`artifacts/runs/generation/`. It refuses to overwrite any existing file; SmolVLM files are untouched.
