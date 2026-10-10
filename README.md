# Multimodal Visual RAG

A retrieval-augmented question answering system over visually rich documents (page images with text, tables, charts and diagrams), built so that every stage can be measured: page retrieval, evidence selection, grounded generation with page citations, and abstention when the evidence is missing.

The system does not send an image to a vision-language model and display the reply. Retrieval and generation are separate components with their own metrics, and each retrieval or generation strategy is a YAML file that can be swapped and compared on identical evaluation examples.

## Overview

Questions about documents are often answerable only from a specific page, and frequently from a table or figure on that page. Three families of page representation compete for that retrieval task:

- text extracted from the page by OCR, searched lexically (BM25) or with dense text embeddings;
- a caption of the page written by a vision-language model, searched as text;
- embeddings of the page image itself, either one vector per page (CLIP, SigLIP) or one vector per image patch with late interaction (ColPali family).

The project implements all three, their hybrids, and a cross-encoder reranker, then feeds the selected pages to a vision-language model in one of four evidence modes (no retrieval, OCR text, page images, images plus text) and scores the answers for correctness, faithfulness, citation accuracy and abstention.

### Research questions

| | Question |
|---|---|
| RQ1 | Does image-text retrieval outperform text-only retrieval for visually grounded questions? |
| RQ2 | How do OCR-based, caption-based and native visual embeddings compare? |
| RQ3 | Does hybrid multimodal retrieval improve recall and answer correctness? |
| RQ4 | Does reranking improve evidence selection enough to justify its cost? |
| RQ5 | How do different vision-language models handle unsupported questions and missing visual evidence? |
| RQ6 | What is the accuracy, latency and memory trade-off between local and GPU inference? |

## Architecture

```
files (PDF, image)                      question
        |                                   |
   page rendering                     +-----+------+
        |                             |            |
   +----+-----------+--------+   lexical index  visual index
   |                |        |   (BM25)         (SigLIP / CLIP / ColPali)
  OCR text     VLM caption  page image          dense text index (bge)
   |                |        |        |            |
   +-------- text sources ---+        +--- fusion (RRF) ---+
                                                |
                                     cross-encoder rerank (optional)
                                                |
                                     evidence selection (page and token budget)
                                                |
                                     VLM generation, pages labelled P1..Pn
                                                |
                              answer + page citations, or INSUFFICIENT_EVIDENCE
```

Design points:

- **One corpus layout.** A corpus is a directory with a page manifest, page images, and one file per text source (`ocr_tesseract`, `parser_markdown`, `caption_<model>`). A strategy names the text source it indexes, so OCR, parser text and captions are compared by changing one field.
- **Deterministic index cache.** Each index directory is addressed by a hash of its parameters, the corpus fingerprint and the indexed text, so a changed corpus can never be served by a stale index.
- **Shared generator across evidence modes.** Text RAG and image RAG use the same vision-language model and the same retrieved pages; only the modality handed to the model differs.
- **Citations resolve to pages.** The generator sees only the labels P1..Pn. Labels are mapped back to page identifiers after generation, and labels that were never supplied are reported as invalid citations.
- **Recognition is separate from grounding.** Describing an uploaded image (`/recognize`) uses no corpus and returns no citations. Grounded answers (`/ask`) always come with the pages shown to the model.

See [docs/architecture.md](docs/architecture.md) for module-level detail.

## Data

The evaluation corpus is the computer science domain of **ViDoRe v3** (Loison et al., 2026): 1,360 page images from two OpenStax textbooks, with graded, human-verified page relevance judgements, reference answers, and a label for the modality of the evidence (text, table, chart, infographic, image). The dataset and the source textbooks are both CC-BY-4.0.

| | |
|---|---|
| Pages | 1,360 (945 + 415 across two textbooks) |
| English queries | 215 (110 human-written, 105 synthetic with human verification) |
| Relevant pages per query | 4.9 on average (median 4, maximum 21), graded 1 or 2 |
| Queries with non-text gold evidence | 119 |
| Text-only queries | 96 |

This benchmark was chosen because it provides real page-level relevance judgements. Document VQA sets such as DocVQA and ChartQA pair each question with a single image and do not state which other pages are irrelevant, so they cannot score retrieval without additional assumptions. [docs/datasets.md](docs/datasets.md) records the licence, access and suitability check for each dataset considered.

No benchmark asset is stored in this repository. `mmrag fetch-vidore` downloads the data from Hugging Face into a Git-ignored directory.

Leakage controls: captions are generated from the page image and a fixed prompt only; queries, reference answers and relevance labels are stored in files that no indexing or captioning code reads; the generator never sees page identifiers.

## Evaluation

**Retrieval.** Recall@K (the fraction of annotated relevant pages retrieved, which is page-level evidence recall), Hit@K, full-evidence Recall@K (all relevant pages retrieved), MRR@10 and nDCG@K with linear gain, matching the `ndcg_cut` definition used by the ViDoRe leaderboards. Every metric is reported with a 95% bootstrap interval, and strategies are compared with a paired bootstrap over the same queries. Results are stratified by whether the gold evidence is textual or visual.

**Generation.** Each system is run under three conditions built from the same questions:

| Condition | Construction | Correct behaviour |
|---|---|---|
| Answerable | normal retrieval over the full corpus | answer and cite |
| Evidence withheld | every annotated relevant page removed from the candidate pool | abstain |
| Out of domain | questions written for a different corpus (ViDoRe v3 HR) | abstain |

Metrics: token F1 and ROUGE-L against the reference answer, yes/no accuracy on boolean questions, an NLI entailment proxy for correctness, NLI faithfulness of the answer to the text of the supplied pages, citation precision and recall against the relevance judgements, abstention rate per condition, and the unsupported-answer rate. The mean token log-probability is evaluated as a confidence signal by its AUROC for separating answers given real evidence from answers given none.

**Systems.** Per-stage latency, peak resident memory, accelerator memory, parameter count and input/output token counts.

The full protocol and the known weaknesses of each automatic metric are in [docs/evaluation.md](docs/evaluation.md).

## Results

Results below are for the text-based retrieval strategies on the full corpus (1,360 pages, 215 queries), run on an Apple M3 Pro with 18 GB of memory. Brackets are 95% bootstrap intervals. The complete tables, including paired comparisons, are in [results/SUMMARY.md](results/SUMMARY.md). The status of every other experiment, with the exact command for each, is in [docs/experiments.md](docs/experiments.md).

| Strategy | R@1 | R@5 | R@10 | nDCG@10 | MRR@10 | Full R@10 | nDCG@10 visual | nDCG@10 text-only | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| BM25 + dense OCR (RRF) + cross-encoder rerank | 0.272 | 0.600 | 0.719 | 0.689 [0.653, 0.725] | 0.845 | 0.433 | 0.651 | 0.737 | 467 |
| BM25 + dense OCR (RRF) | 0.284 | 0.584 | 0.713 | 0.689 [0.652, 0.725] | 0.841 | 0.433 | 0.658 | 0.728 | 38 |
| Dense text over OCR (bge-small) | 0.268 | 0.561 | 0.686 | 0.663 [0.627, 0.697] | 0.831 | 0.381 | 0.630 | 0.704 | 33 |
| Dense text over dataset parser text | 0.256 | 0.558 | 0.662 | 0.648 [0.613, 0.682] | 0.823 | 0.349 | 0.633 | 0.667 | 32 |
| BM25 over dataset parser text | 0.267 | 0.526 | 0.650 | 0.635 [0.597, 0.675] | 0.804 | 0.377 | 0.610 | 0.666 | 2.3 |
| BM25 over OCR | 0.265 | 0.525 | 0.651 | 0.633 [0.595, 0.673] | 0.809 | 0.372 | 0.605 | 0.669 | 2.4 |

Findings supported by these runs:

- **Hybrid text retrieval is the strongest text configuration.** Fusing BM25 and dense rankings raises nDCG@10 from 0.663 (dense alone) to 0.689, a paired difference of +0.026 [+0.001, +0.050], p = 0.046. The gain over BM25 alone is +0.056.
- **Dense and lexical retrieval are not separable at this sample size.** Dense minus BM25 is +0.030 [-0.006, +0.066], p = 0.11.
- **Cross-encoder reranking does not pay for itself here.** Reranking the top 20 with an MS MARCO MiniLM cross-encoder changes nDCG@10 by +0.001 [-0.024, +0.025] while raising latency from 38 ms to 467 ms per query. On queries with visual evidence the point estimate is slightly negative (-0.006), consistent with a text-only reranker being unable to see tables and figures.
- **OCR quality is not the bottleneck on this corpus.** Tesseract text and the parser text shipped with the dataset give the same BM25 quality (difference +0.002 [-0.007, +0.010]). The pages are clean born-digital renders, so this should not be assumed for scanned documents.
- **Visually grounded questions are harder for every text strategy.** nDCG@10 is 0.03 to 0.09 lower on queries whose gold evidence includes a table, chart, infographic or image than on text-only queries.
- **Complete evidence sets are rarely retrieved.** Even the best strategy places all relevant pages in the top 10 for only 43% of queries, which bounds what a generator restricted to a few pages can cite.

Latency figures were measured on a machine shared with other workloads and should be read as indicative.

## Installation

Requires Python 3.11 or 3.12 and, for the OCR baselines, the Tesseract binary.

```bash
brew install tesseract            # macOS; apt-get install tesseract-ocr on Debian/Ubuntu
uv venv --python 3.12
uv pip install -e ".[models,app,dev]"
```

Optional extras: `late` (ColPali-family retrievers through colpali-engine) and `quant` (4-bit loading with bitsandbytes, CUDA only). Without the `models` extra the package still provides ingestion, OCR, BM25 retrieval, the API and the UI with a deterministic mock generator.

If downloads hang on a network with a broken IPv6 route, set `MMRAG_FORCE_IPV4=1`.

## Usage

```bash
# 1. Data: download the benchmark, run OCR, fetch the out-of-domain probe questions
mmrag fetch-vidore --domain computer_science --drop-raw
mmrag ocr -c data/processed/vidore_v3_computer_science
mmrag fetch-ood -c data/processed/vidore_v3_computer_science

# 2. Retrieval benchmark (indexes are built on first use and cached)
mmrag eval-retrieval configs/retrieval/bm25_ocr.yaml configs/retrieval/hybrid_text.yaml \
    -c data/processed/vidore_v3_computer_science

# 3. Generation benchmark: same questions and evidence for every system
mmrag eval-generation configs/generation/smolvlm_500m_text_rag.yaml \
    configs/generation/smolvlm_500m_image_rag.yaml \
    -c data/processed/vidore_v3_computer_science \
    --retrieval configs/retrieval/hybrid_text.yaml --n-queries 60 --oracle

# 4. Tables
mmrag report

# Ask a single question from the command line
mmrag ask "How does the V-model integrate testing?" -c data/processed/vidore_v3_computer_science
```

Your own documents:

```bash
mmrag ingest report.pdf figure.png -c uploads/corpus     # renders pages and runs OCR
```

### Application

```bash
uvicorn mmrag.api.app:app --port 8000        # API, interactive docs at /docs
streamlit run app/streamlit_app.py           # UI on http://localhost:8501
```

The UI has three tabs: **Ask** (grounded answer, cited pages, the evidence shown to the model, uncertainty indicators, timings), **Search** (retrieval only, by text or by image) and **Recognise an image** (model description and OCR of one upload, with no corpus involved). Documents uploaded in the sidebar become a searchable corpus.

The generator and default strategy are chosen with environment variables, for example `MMRAG_GENERATION=smolvlm_500m_image_rag MMRAG_STRATEGY=hybrid_text`. The default generator is `mock`, which loads no model.

With Docker:

```bash
docker compose up --build
```

### GPU experiments

`kaggle/run_gpu.py` runs the larger models (ColQwen2 retrieval, Qwen3-VL 4B and 8B generation, full-corpus captioning, and the ablations over page count, image resolution and text budget) on a Kaggle T4. See [kaggle/README.md](kaggle/README.md).

## Configuration

A retrieval strategy:

```yaml
name: hybrid_text_colsmol_rerank
retriever:
  type: hybrid
  fusion: rrf
  components:
    - {type: bm25, text_source: ocr_tesseract}
    - {type: dense_text, text_source: ocr_tesseract}
    - {type: late_interaction, model: vidore/colSmol-256M}
rerank: {type: cross_encoder, model: cross-encoder/ms-marco-MiniLM-L6-v2, top_n: 20}
```

A generation system:

```yaml
name: qwen3vl_2b_image_rag
vlm: {model: Qwen/Qwen3-VL-2B-Instruct}
evidence: {mode: image, max_pages: 3, image_max_side: 1024}
```

Unknown keys are rejected at load time.

## Models

All models are openly licensed and run without a paid API.

| Role | Model | Parameters | Licence |
|---|---|---|---|
| Dense text | BAAI/bge-small-en-v1.5 | 33 M | MIT |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 | 23 M | Apache-2.0 |
| Visual, single vector | google/siglip2-base-patch16-224 | 375 M | Apache-2.0 |
| Visual, late interaction | vidore/colSmol-256M | 256 M | MIT |
| Visual, late interaction (GPU) | vidore/colqwen2-v1.0 | 2.2 B | Apache-2.0 |
| Generator, local | HuggingFaceTB/SmolVLM-500M-Instruct | 507 M | Apache-2.0 |
| Generator, local | Qwen/Qwen3-VL-2B-Instruct | 2.1 B | Apache-2.0 |
| Generator, GPU | Qwen/Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct (4-bit) | 4.4 B, 8 B | Apache-2.0 |
| Faithfulness scoring | cross-encoder/nli-deberta-v3-xsmall | 22 M | Apache-2.0 |

## Repository structure

```
src/mmrag/
  data/        corpus layout, ViDoRe v3 materialisation, pool-corpus builder
  ingest/      PDF and image parsing, Tesseract OCR, VLM captioning
  index/       BM25, dense text, single-vector visual, late-interaction indexes
  retrieve/    fusion, cross-encoder reranking, configurable pipeline
  generate/    VLM backends, prompts, citation parsing, RAG pipeline
  eval/        retrieval and generation metrics, bootstrap statistics, harness, report
  api/         FastAPI service
  cli.py       command-line interface
app/           Streamlit front end
configs/       retrieval strategies and generation systems
kaggle/        GPU runner
tests/         unit and API tests
docs/          architecture, datasets, evaluation protocol, research audit, experiments
results/       metric files and summary tables
```

## Tests

```bash
HF_HUB_OFFLINE=1 pytest
```

The suite builds a small synthetic corpus from generated images and uses the mock generator, so it needs no network access, GPU or downloaded model. It covers the metrics against hand-computed values, fusion, index caching, evidence withholding, citation parsing, the three-condition evaluation harness, ingestion and the API.

## Limitations

- **One domain, modest sample.** 215 English queries over two textbooks from one publisher. Differences smaller than about 0.03 nDCG@10 are within the noise, and the pages are clean born-digital renders, which favours OCR.
- **Automatic answer metrics.** Reference answers are free-form sentences. Token F1 and ROUGE-L penalise correct paraphrases, and the NLI proxies use a small model. No LLM judge or human grading is included.
- **Faithfulness is measured against page text.** A correct claim supported only by a chart or diagram is counted as unsupported.
- **Text-only reranker.** The reranker cannot use visual evidence. A visual reranker is not implemented.
- **Exact search.** Indexes are scored exhaustively in NumPy or on the accelerator, which is appropriate for thousands of pages and not for millions.
- **Uncalibrated uncertainty.** The log-probability and retrieval-margin indicators shown in the UI are raw signals, not probabilities.
- **Synthetic queries.** About half of the benchmark queries were generated by a model and then verified by annotators.

## Possible extensions

- Additional ViDoRe v3 domains (finance, pharmaceuticals, industrial) and scanned-document sets where OCR is unreliable.
- A visual reranker that scores query and page image jointly.
- Region-level evidence using the bounding boxes provided with the relevance judgements.
- Approximate nearest-neighbour search and token pooling for late-interaction indexes at larger scale.
- An LLM judge validated against human grades for answer correctness.

## References

- Loison et al., "ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios", arXiv:2601.08620, 2026.
- Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models", ICLR 2025.
- Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features", 2025.
- Marafioti et al., "SmolVLM: Redefining Small and Efficient Multimodal Models", arXiv:2504.05299, 2025.
- Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods", SIGIR 2009.
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.

The project idea follows "Chatbot with Image Recognition", project 15 in a public collection of generative AI project ideas by Bhavishya Pandit.

## License

Code: MIT, see [LICENSE](LICENSE). The benchmark data is distributed by its authors under CC-BY-4.0 and is not included in this repository.
