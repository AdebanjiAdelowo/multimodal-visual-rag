# Experiment status

Last updated: 2026-10-11. A row is marked DONE only if its result file is in `results/`.

`C=data/processed/vidore_v3_computer_science` in all commands. Set `MMRAG_FORCE_IPV4=1` on networks with a broken IPv6 route.

## Data preparation

| Step | Status | Notes |
|---|---|---|
| Materialise ViDoRe v3 computer science | DONE | 1,360 pages, 215 English queries, 1,049 qrels, revision `d5cc7588` |
| Tesseract OCR | DONE | 1,360 pages in 5 min 6 s with 8 workers; 2 pages have no text |
| Out-of-domain probe queries | DONE | 318 English queries from the ViDoRe v3 HR domain (query text only) |
| Page captions | PENDING (GPU) | see below |

## Retrieval (full corpus)

| Strategy | Status | Result |
|---|---|---|
| `bm25_ocr`, `bm25_parser`, `dense_ocr`, `dense_parser`, `hybrid_text`, `hybrid_text_rerank` | DONE | `results/retrieval/vidore_v3_computer_science/` |
| `visual_siglip`, `hybrid_bm25_siglip`, `hybrid_text_siglip`, `hybrid_text_siglip_w025`, `hybrid_text_siglip_rerank` | DONE | same directory; run on 2026-10-10 with lowercased queries |
| `late_colsmol`, `hybrid_text_colsmol`, `hybrid_text_colsmol_rerank` | DONE | same directory; the page index took 29 minutes to build on MPS (1,139 patch vectors per page, 398 MB on disk) |
| `visual_clip` | PENDING (GPU) | Kaggle stage `retrieval` (0.6 GB download). |
| `late_colqwen2` | PENDING (GPU) | Kaggle stage `retrieval`. |

Commands:

```bash
# SigLIP strategies (index cached; about 1.5 GB of accelerator memory, 1 to 2 minutes)
mmrag eval-retrieval configs/retrieval/{visual_siglip,hybrid_bm25_siglip,hybrid_text_siglip,hybrid_text_siglip_w025,hybrid_text_siglip_rerank}.yaml -c $C

# ColSmol strategies (index cached; needs the `late` extra; about 3.4 GB of accelerator memory)
mmrag eval-retrieval configs/retrieval/{late_colsmol,hybrid_text_colsmol,hybrid_text_colsmol_rerank}.yaml -c $C

# Clean latency pass on an idle machine: re-run every strategy, then rebuild the tables
mmrag eval-retrieval configs/retrieval/*.yaml -c $C
mmrag report
```

A SigLIP run made before query lowercasing was added (nDCG@10 0.087 alone, 0.550 in the equal-weight hybrid) is kept in `artifacts/preliminary/` and is not reported; the committed results (0.094 and 0.556) replace it.

## Caption-based retrieval

Status: PENDING (GPU). Captioning was measured at 15 to 19 s per page with SmolVLM-500M on the shared local machine, which is about six hours for the full corpus, so both captioners run on the GPU over all 1,360 pages (Kaggle stage `captions`). OCR, captions and visual embeddings are then compared on the full corpus and the full query set, in the same result directory.

```bash
# Kaggle (see kaggle/README.md)
MMRAG_STAGES=data,retrieval,captions
```

A 400-page pool corpus for a smaller local version of this experiment can be built with `mmrag make-pool -c $C --n-queries 60 --n-pages 400`; it keeps every relevant page of 60 sampled queries (245 pages) plus random distractors. It is not used for any reported result.

## Generation

| Experiment | Status | Where |
|---|---|---|
| SmolVLM-500M, four evidence modes + oracle, 60 answerable, 60 evidence-withheld, 30 out-of-domain | DONE (local, MPS) | `results/generation/vidore_v3_computer_science/` |
| Same design on the T4 with SmolVLM-500M, Qwen3-VL-2B, Qwen3-VL-4B | PENDING (GPU) | Kaggle stage `generation` |
| Qwen3-VL-8B in 4-bit, same design | PENDING (GPU) | Kaggle stage `generation_8b` |

Local command:

```bash
mmrag eval-generation configs/generation/smolvlm_500m_{closed_book,text_rag,image_rag,image_text_rag}.yaml -c $C --retrieval configs/retrieval/hybrid_text.yaml --n-queries 60 --n-ood 30 --oracle
```

Notes on the local run. It took about 20 minutes of generation and about 25 minutes of NLI scoring on the CPU. The first attempt was stopped after 181 generations because the MPS buffer cache had grown to 15.4 GB; it was resumed after the fix in `TransformersVLM.generate`. Closed-book and text-RAG were then repeated from scratch: all 240 answers were identical, and the text-RAG records were replaced by the repeat so that their latency is not affected by the swapping of the first attempt.

Qwen3-VL-2B is cached locally (4.0 GB) and would fit in memory, but it runs on the GPU together with the larger models so that all generators are measured on one machine.

## Ablations

| Ablation | Status | Where |
|---|---|---|
| OCR vs dataset parser text | DONE | `bm25_ocr` vs `bm25_parser`, `dense_ocr` vs `dense_parser` |
| Lexical vs dense vs hybrid text | DONE | results summary |
| Reranking on/off (text hybrid) | DONE | results summary |
| OCR vs captions | PENDING (GPU) | Kaggle stage `captions` |
| Text vs visual embeddings; single-vector vs late interaction | DONE | `dense_ocr` vs `visual_siglip` vs `late_colsmol` |
| Fusion weight of the visual component | DONE | `hybrid_text_siglip` vs `hybrid_text_siglip_w025` |
| Reranking on/off (multimodal hybrid) | DONE | `hybrid_text_colsmol` vs `hybrid_text_colsmol_rerank`, `hybrid_text_siglip` vs `hybrid_text_siglip_rerank` |
| Number of pages given to the generator | PENDING (GPU) | `--max-pages 1`, `2`, `5` on `eval-generation` |
| Image resolution | PENDING (GPU) | `--image-max-side 512`, `768`, `1536` |
| Text context budget | PENDING (GPU) | `--max-chars 800`, `5000` |
| Direct model vs RAG | DONE for SmolVLM-500M | `*_closed_book` vs `*_rag` systems; larger models on GPU |
| Small vs larger generators | PENDING (GPU) | SmolVLM-500M done locally; Qwen3-VL 2B, 4B, 8B on GPU |
| Query origin (human vs synthetic) | DONE | origin table in the results summary |
| Qualitative failure cases with source references | PENDING | to be drawn from `artifacts/runs/generation/*.scored.jsonl` once the larger generators have run |

## Why runs are pending

The remaining experiments need a CUDA GPU: the 4B and 8B generators and ColQwen2 do not fit in 18 GB of unified memory alongside other work, and full-corpus captioning takes hours locally. `kaggle/run_gpu.py` runs all of them and writes to `results_t4/`, leaving the local `results/` untouched.
