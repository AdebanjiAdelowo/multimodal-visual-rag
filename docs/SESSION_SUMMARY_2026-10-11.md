# Session summary, 2026-10-11

End-of-session report for session 3 (2026-10-10 to 2026-10-11). To resume, read this file, then [KAGGLE_HANDOFF.md](KAGGLE_HANDOFF.md) for the Kaggle steps and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the status of every area.

**State at the end of the session:** local work is finished and committed (`main` at `3eb6fdb` plus the commit that adds this file; no remote, nothing pushed). The Kaggle job is **not running**: Kaggle refused the kernel push twice because both GPU batch slots on the account were held by other projects' jobs (`legal-research-rag-gpu` running, `research-innovation-monitor-gpu` queued). Those were left alone.

## 1. Where the project stands

- **Retrieval:** complete locally for 14 strategies on the full corpus (1,360 pages, 215 queries).
- **Generation:** complete locally for SmolVLM-500M (60 answerable, 60 evidence-withheld, 30 out-of-domain).
- **Waiting on GPU:** CLIP, ColQwen2, caption-based retrieval, Qwen3-VL 2B/4B/8B, generation ablations.
- **Never executed:** Docker build and CI. The Streamlit app has no automated test.

## 2. Previous stopping point

Session 2 ended at `a59df54` with a clean tree and all code committed; only text retrieval had been run.

## 3. What changed this session

- **Environment repair:** iCloud had evicted files inside the repo (including `.git` objects), which made `git status` hang; they were re-downloaded. macOS also keeps re-hiding the venv's `.pth` files, so commands need `PYTHONPATH=src`.
- **Memory bug:** the SmolVLM run's MPS cache grew to 15.4 GB. After the stop-fix-resume decision it holds at about 3 GB.
- **Resumable ColSmol index:** the 29-minute build now checkpoints in shards.
- **Report additions:** human vs synthetic query breakdown, and paired significance tests between generation modes.
- **Kaggle runner:** it writes to `results_t4/`, so local results cannot be overwritten. It rewrites a checksummed archive after every command, stops starting new work after 10.5 hours, and records the source commit. Its control flow is dry-run tested but it has never run on Kaggle.
- **`pad_token_id` warning:** harmless. It comes from a default config that transformers builds only to print a diff; the loaded model uses pad id 2.
- **Docs:** README results, experiment status, research audit, `docs/IMPLEMENTATION_STATUS.md` and `docs/KAGGLE_HANDOFF.md`.

## 4. Commits (local only, nothing pushed)

| Commit | Content |
|---|---|
| `add3358` | Resumable late-interaction index |
| `a18db67` | SigLIP2 results |
| `41a4640` | ColSmol results |
| `f66d4fc` | MPS memory fix |
| `2b8fd03` | Query-origin strata and paired generation comparisons |
| `1d251d3` | Kaggle runner for captions and Qwen3-VL |
| `434515d` | SmolVLM-500M generation results |
| `564cd25` | Runner hardening for batch execution |
| `3a723f9` | Documentation and single-session retrieval re-measurement |
| `3eb6fdb` | Kaggle handoff and IPv4 client wrapper |

## 5. Tests

42 pass and ruff is clean (`PYTHONPATH=src make test && make lint`).

## 6. Results measured

Full tables are in [../results/SUMMARY.md](../results/SUMMARY.md).

**Retrieval (nDCG@10, 95% interval):**

| Strategy | nDCG@10 | ms/query |
|---|---|---|
| BM25 + dense OCR + ColSmol | 0.695 [0.661, 0.727] | 76 |
| BM25 + dense OCR + rerank | 0.689 [0.653, 0.725] | 271 |
| BM25 + dense OCR | 0.689 [0.652, 0.725] | 14 |
| Dense OCR | 0.663 [0.627, 0.697] | 11 |
| BM25 OCR | 0.633 [0.595, 0.673] | 1.2 |
| ColSmol late interaction | 0.608 [0.568, 0.647] | 66 |
| BM25 + dense OCR + SigLIP2, equal weight | 0.556 [0.525, 0.588] | 23 |
| SigLIP2, one vector per page | 0.094 [0.071, 0.119] | 9.8 |

- **Visual retrieval lost to OCR text on visual questions:** ColSmol 0.572 vs dense OCR 0.630 (-0.057, p = 0.018).
- **Adding ColSmol to the text hybrid is within noise:** +0.007 [-0.011, +0.024].
- **Adding SigLIP2 hurts:** -0.133 at equal weight.
- **Reranking gains nothing on the text hybrid:** +0.001 at about 20 times the latency.

**Generation (SmolVLM-500M, token F1):**

| Evidence | Token F1 | Abstains when evidence withheld |
|---|---|---|
| None | 0.251 | n/a |
| OCR text | 0.343 | 1.7% |
| Page images | 0.299 | 0% |
| Images + text | 0.290 | 0% |
| Gold page images | 0.306 | n/a |

- **Text RAG beats the direct model:** +0.093 [+0.032, +0.159], p = 0.003. Image RAG does not clearly do so (p = 0.20).
- **The model almost never abstains** and cites a page in at most 10% of answers.
- **Reproducibility:** a repeat run reproduced all 240 closed-book and text-RAG answers exactly, and a retrieval re-run reproduced every ranking.

## 7. Still pending

- **Kaggle job:** the source dataset `adebanjiadelowo/multimodal-visual-rag-src` is uploaded (private, commit `3a723f9`). The kernel `multimodal-visual-rag-gpu` does not exist yet.
- **Docker and CI:** unverified.

## 8. Concerns

- **Narrow corpus:** two born-digital textbooks with near-perfect OCR. The visual-vs-text conclusion will not transfer to scans.
- **Small generation sample:** 60 questions give about +/- 0.06 on token F1, so differences between evidence modes are mostly unresolved. Citation metrics rest on four to six answers.
- **Uncorrected p-values:** the BM25 + dense gain (p = 0.046) would not survive correction.
- **One retrieval strategy for generation:** nothing can be claimed yet about hybrid retrieval changing answer correctness.
- **Kaggle time estimates are untested:** the job may not finish all six stages in one run; skipped commands are recorded.
- **Disk:** it dropped to 11 GB free during the session, which is what triggers the iCloud eviction.

## 9. Exact next task

First check whether the kernel already exists and has run; if it shows `COMPLETE`, retrieve its output instead of submitting again.

```bash
cd ~/Documents/GitHub/multimodal-visual-rag
make kaggle-status KAGGLE_PYTHON=/opt/anaconda3/bin/python     # 404 means it was never submitted
```

If it was never submitted and a GPU slot is free:

```bash
make kaggle-submit KAGGLE_PYTHON=/opt/anaconda3/bin/python
make kaggle-status KAGGLE_PYTHON=/opt/anaconda3/bin/python     # expect QUEUED or RUNNING
```

Record the kernel version in `docs/KAGGLE_HANDOFF.md` and stop. When the job finishes, follow the retrieve-and-validate section of that file before importing anything.

Prompt to give the next session:

> Continue `~/Documents/GitHub/multimodal-visual-rag`. Read `docs/SESSION_SUMMARY_2026-10-11.md` and `docs/KAGGLE_HANDOFF.md`, check the Kaggle kernel status once, then either submit, or retrieve and validate the results, as those files describe.
