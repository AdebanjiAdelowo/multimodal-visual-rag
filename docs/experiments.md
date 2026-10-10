# Experiment status

Last updated: 2026-10-10. A row is marked DONE only if its result file is in `results/`.

`C=data/processed/vidore_v3_computer_science` in all commands. Set `MMRAG_FORCE_IPV4=1` on networks with a broken IPv6 route.

## Data preparation

| Step | Status | Notes |
|---|---|---|
| Materialise ViDoRe v3 computer science | DONE | 1,360 pages, 215 English queries, 1,049 qrels, revision `d5cc7588` |
| Tesseract OCR | DONE | 1,360 pages in 5 min 6 s with 8 workers; 2 pages have no text |
| Out-of-domain probe queries | DONE | 318 English queries from the ViDoRe v3 HR domain (query text only) |
| Page captions | PENDING | see below |

## Retrieval (full corpus)

| Strategy | Status | Result |
|---|---|---|
| `bm25_ocr`, `bm25_parser`, `dense_ocr`, `dense_parser`, `hybrid_text`, `hybrid_text_rerank` | DONE | `results/retrieval/vidore_v3_computer_science/` |
| `visual_siglip`, `hybrid_bm25_siglip`, `hybrid_text_siglip`, `hybrid_text_siglip_w025`, `hybrid_text_siglip_rerank` | PENDING (re-run) | The page index is built and cached. An earlier run exists but predates query lowercasing, so it is not reported. |
| `late_colsmol`, `hybrid_text_colsmol`, `hybrid_text_colsmol_rerank` | PENDING | Model weights are cached; the page index was interrupted during its first build. |
| `visual_clip` | PENDING | Needs a 0.6 GB model download. |
| `late_colqwen2` | PENDING (GPU) | Kaggle stage `retrieval`. |

Commands:

```bash
# SigLIP strategies (index cached; about 1.5 GB of accelerator memory, 1 to 2 minutes)
mmrag eval-retrieval configs/retrieval/{visual_siglip,hybrid_bm25_siglip,hybrid_text_siglip,hybrid_text_siglip_w025,hybrid_text_siglip_rerank}.yaml -c $C

# ColSmol strategies (builds a late-interaction index for 1,360 pages; needs the `late` extra)
mmrag eval-retrieval configs/retrieval/{late_colsmol,hybrid_text_colsmol,hybrid_text_colsmol_rerank}.yaml -c $C

# Clean latency pass on an idle machine: re-run every strategy, then rebuild the tables
mmrag eval-retrieval configs/retrieval/{bm25_ocr,bm25_parser,dense_ocr,dense_parser,hybrid_text,hybrid_text_rerank}.yaml -c $C
mmrag report
```

Superseded development observation (not a reported result): before query lowercasing, single-vector SigLIP2 base at 224 px reached nDCG@10 of 0.087 on the full corpus, and adding it to the text hybrid with equal RRF weight lowered nDCG@10 from 0.689 to 0.550. Lowercasing raised the SigLIP figure to 0.094 in a direct check. The pending re-run will replace these numbers.

## Caption-based retrieval

Captioning all 1,360 pages with SmolVLM-500M was measured at 15 to 19 s per page on the shared local machine, so local caption experiments use a 400-page pool corpus (60 sampled queries, all of their relevant pages, random distractors). Every strategy is evaluated on the same pool for a like-for-like comparison. Full-corpus captions run on the GPU.

```bash
mmrag make-pool -c $C --n-queries 60 --n-pages 400
P=data/processed/vidore_v3_computer_science_pool400
mmrag caption -c $P --config configs/generation/caption_smolvlm_500m.yaml
mmrag eval-retrieval configs/retrieval/{bm25_ocr,dense_ocr,bm25_caption,dense_caption,visual_siglip,late_colsmol,hybrid_text}.yaml -c $P
```

Status: PENDING.

## Generation

Smoke test only (one question, gold pages, SmolVLM-500M on MPS): the model loaded in float16 (968 MB of weights), produced answers in 3.5 to 4.9 s, and ignored the `ANSWER:` / `CITATIONS:` format in both RAG modes; in closed-book mode it abstained. This is an observation from a single example, not a result.

| Experiment | Status | Command |
|---|---|---|
| SmolVLM-500M, four evidence modes + oracle, 60 queries, 30 out-of-domain | PENDING | `mmrag eval-generation configs/generation/smolvlm_500m_{closed_book,text_rag,image_rag,image_text_rag}.yaml -c $C --retrieval configs/retrieval/hybrid_text.yaml --n-queries 60 --n-ood 30 --oracle` |
| Qwen3-VL-2B, same design (weights cached, 4.3 GB; about 5 GB of accelerator memory) | PENDING | `mmrag eval-generation configs/generation/qwen3vl_2b_{closed_book,text_rag,image_rag,image_text_rag}.yaml -c $C --retrieval configs/retrieval/hybrid_text.yaml --n-queries 60 --n-ood 30 --oracle` |
| Qwen3-VL-4B and 8B (4-bit), all queries | PENDING (GPU) | Kaggle stage `generation` |

## Ablations

| Ablation | Status | Where |
|---|---|---|
| OCR vs dataset parser text | DONE | `bm25_ocr` vs `bm25_parser`, `dense_ocr` vs `dense_parser` |
| Lexical vs dense vs hybrid text | DONE | results summary |
| Reranking on/off (text hybrid) | DONE | results summary |
| OCR vs captions | PENDING | pool corpus above; full corpus on GPU |
| Text vs visual embeddings; single-vector vs late interaction | PENDING | SigLIP and ColSmol commands above |
| Fusion weight of the visual component | PENDING | `hybrid_text_siglip` vs `hybrid_text_siglip_w025` |
| Reranking on/off (multimodal hybrid) | PENDING | `hybrid_text_colsmol` vs `hybrid_text_colsmol_rerank` |
| Number of pages given to the generator | PENDING | `--max-pages 1`, `2`, `5` on `eval-generation` |
| Image resolution | PENDING | `--image-max-side 512`, `768`, `1536` |
| Text context budget | PENDING | `--max-chars 800`, `5000` |
| Direct model vs RAG | PENDING | `*_closed_book` vs `*_rag` systems |
| Small vs larger generators | PENDING | SmolVLM-500M, Qwen3-VL-2B locally; 4B and 8B on GPU |
| Qualitative failure cases with source references | PENDING | to be drawn from `artifacts/runs/generation/*.scored.jsonl` after the runs |

## Why runs are pending

The local machine is shared with other workloads. At the time of the last check swap usage was above 9 GB and the load average was near 20, so accelerator experiments were not started. All pending local runs fit in memory on an idle 18 GB machine.
