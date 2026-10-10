# Architecture

## Data flow

```
ingest/parse.py     PDF or image  ->  page JPEGs + pages.jsonl
ingest/ocr.py       page image    ->  text/ocr_tesseract.jsonl
ingest/caption.py   page image    ->  text/caption_<model>.jsonl
index/*             text source or page images  ->  cached index
retrieve/pipeline   query -> per-index scores -> fusion -> (rerank) -> ranked pages
generate/rag.py     ranked pages -> evidence selection -> prompt -> VLM -> parsed answer
eval/*              rankings and answers -> metrics -> results/*.json -> results/SUMMARY.md
```

## Corpus layout

```
<corpus>/
  pages.jsonl            page_id, doc_id, page_number, image_path, width, height
  images/<page_id>.jpg
  text/<source>.jsonl    page_id -> text
  queries.jsonl          evaluation only
  qrels.jsonl            evaluation only
  ood_queries.jsonl      evaluation only
  meta.json              dataset, revision, licence
```

A page is the retrieval unit throughout. `Corpus.load_text` refuses a text source that does not cover every page, so an index can never silently omit pages.

## Indexes

Every index implements `score_all(query) -> array` with one score per page in manifest order. Ranking, exclusion, fusion and reranking are generic code on top of that array.

| Index | Representation | Query side | Page score |
|---|---|---|---|
| `BM25Index` | tokens of one text source | tokenised query | Okapi BM25 (k1 = 1.5, b = 0.75) |
| `DenseTextIndex` | sentence-transformer embeddings of 180-word chunks, 40-word overlap | embedded query with the model's retrieval prefix | maximum cosine over the page's chunks |
| `VisualIndex` | one CLIP or SigLIP embedding per page image | text tower (or image tower for query by image) | cosine |
| `LateInteractionIndex` | one embedding per image patch (ColPali family) | one embedding per query token | MaxSim: sum over query tokens of the best patch similarity |

Search is exhaustive. At 1,360 pages a dense matrix product is faster than any approximate index, and exact scores remove one source of variance from the comparison.

Index artefacts live under `artifacts/index/<corpus>/<kind>__<label>__<hash>/`. The hash covers the index parameters, the corpus fingerprint (page ids, image paths, sizes) and, for text indexes, the text itself. `meta.json` is written last and marks the directory complete.

`IndexRegistry` builds each index once per process and shares it, so a hybrid strategy reuses the indexes of its single-retriever baselines.

## Fusion and reranking

- **Reciprocal rank fusion** (default): `sum_i w_i / (k + rank_i)`, k = 60. Rank-based, so BM25 scores and cosine similarities need no calibration. Component weights are configurable.
- **z-score fusion**: weighted sum of standardised scores.
- **Cross-encoder reranking**: the top N pages are rescored on their best text chunk; pages beyond N keep their first-stage order below the reranked block.

`RetrievalPipeline.search(query, k, exclude)` accepts a set of page ids to remove from the candidate pool before ranking. The evaluation uses it to build the missing-evidence condition.

## Generation

`VLM.generate(content)` takes a list of text and image parts. `TransformersVLM` wraps any chat-template image-text-to-text model; image resolution is controlled per call (tile count for SmolVLM-style processors, pre-resizing otherwise). Decoding is greedy. The mean token log-probability is computed from the generation scores. `MockVLM` is a deterministic stand-in for tests.

`select_evidence` keeps the top `max_pages` pages, labels them P1..Pn and truncates text to `max_chars_per_page`. `build_prompt` lays out the evidence by mode and appends fixed rules. `parse_answer` extracts the answer, the abstention flag and the citations, and maps labels back to page ids.

`RAGPipeline.answer` can be given precomputed hits, which is how the evaluation supplies identical evidence to every system and how the oracle variant supplies gold pages.

## Device and memory

`pick_device` resolves `auto` to CUDA, then MPS, then CPU. Models load in float16 on accelerators and float32 on CPU; 4-bit NF4 loading through bitsandbytes is available on CUDA. One generator is held in memory at a time during evaluation.

| Model | Weights on disk | Local (M3 Pro, 18 GB) | Kaggle T4 (16 GB) |
|---|---|---|---|
| bge-small, MiniLM cross-encoder | < 0.3 GB | yes | yes |
| SigLIP2 base | 1.5 GB | yes | yes |
| colSmol-256M | 0.5 GB | yes | yes |
| ColQwen2 | 8.8 GB (fp32 on disk) | too large alongside other work | yes, fp16 |
| SmolVLM-500M | 1.0 GB | yes | yes |
| Qwen3-VL-2B | 4.3 GB | yes, fp16 | yes |
| Qwen3-VL-4B | 8.9 GB | no | yes, fp16 |
| Qwen3-VL-8B | 17.5 GB | no | yes, 4-bit (about 6 GB) |

## Application

`api/service.py` owns corpora, registries, pipelines and the generator, all loaded lazily. `api/app.py` exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health`, `/corpora`, `/strategies` | discovery |
| `POST /search` | text query to ranked pages with component scores and stage timings |
| `POST /search/image` | image query to visually similar pages |
| `POST /recognize` | description and OCR of one image; no corpus, no citations |
| `POST /ask` | grounded answer, citations, evidence shown, uncertainty indicators, timings |
| `POST /upload` | add PDFs or images to the upload corpus |
| `GET /pages/{corpus}/{page_id}/image`, `/text` | cited page content |

Page paths are taken from the manifest entry for a validated page id, never from the request string. Uploaded filenames contribute only a sanitised stem and a whitelisted suffix. Uploads are limited to 25 MB per file.

The Streamlit front end is a thin client of the API.
