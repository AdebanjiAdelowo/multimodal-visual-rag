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
