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

Retrieval and SmolVLM numbers below were measured on an Apple M3 Pro with 18 GB of memory. Qwen3-VL-2B generation ran on a Colab Tesla T4; scoring ran locally on CPU. Brackets are 95% bootstrap intervals. The complete tables, including every paired comparison, are generated from the result files into [results/SUMMARY.md](results/SUMMARY.md). Experiments that need a CUDA GPU (ColQwen2, caption-based retrieval, larger generators) are listed with their commands in [docs/experiments.md](docs/experiments.md).

### Retrieval

Full corpus: 1,360 pages, 215 queries.

| Strategy | R@1 | R@5 | R@10 | nDCG@10 | MRR@10 | Full R@10 | nDCG@10 visual | nDCG@10 text-only | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| BM25 + dense OCR + ColSmol (RRF) | 0.272 | 0.604 | 0.726 | 0.695 [0.661, 0.727] | 0.845 | 0.428 | 0.662 | 0.737 | 76 |
| BM25 + dense OCR (RRF) + cross-encoder rerank | 0.272 | 0.600 | 0.719 | 0.689 [0.653, 0.725] | 0.845 | 0.433 | 0.651 | 0.737 | 271 |
| BM25 + dense OCR (RRF) | 0.284 | 0.584 | 0.713 | 0.689 [0.652, 0.725] | 0.841 | 0.433 | 0.658 | 0.728 | 14 |
| BM25 + dense OCR + SigLIP2 at weight 0.25 (RRF) | 0.247 | 0.577 | 0.713 | 0.666 [0.631, 0.698] | 0.810 | 0.419 | 0.644 | 0.694 | 23 |
| Dense text over OCR (bge-small) | 0.268 | 0.561 | 0.686 | 0.663 [0.627, 0.697] | 0.831 | 0.381 | 0.630 | 0.704 | 11 |
| BM25 over OCR | 0.265 | 0.525 | 0.651 | 0.633 [0.595, 0.673] | 0.809 | 0.372 | 0.605 | 0.669 | 1.2 |
| ColSmol-256M late interaction (page images only) | 0.241 | 0.521 | 0.644 | 0.608 [0.568, 0.647] | 0.750 | 0.372 | 0.572 | 0.653 | 66 |
| BM25 + dense OCR + SigLIP2 at equal weight (RRF) | 0.168 | 0.459 | 0.664 | 0.556 [0.525, 0.588] | 0.687 | 0.353 | 0.532 | 0.586 | 23 |
| SigLIP2 base, 224 px, one vector per page | 0.034 | 0.072 | 0.103 | 0.094 [0.071, 0.119] | 0.176 | 0.023 | 0.084 | 0.107 | 9.8 |

Findings, with paired bootstrap differences in nDCG@10 over the same queries:

- **Image-based retrieval did not beat text retrieval on visually grounded questions (RQ1).** On the 119 queries whose evidence includes a table, chart, infographic or image, ColSmol late interaction scores 0.572 against 0.630 for dense OCR text (-0.057 [-0.111, -0.010], p = 0.018). The pages are born-digital textbook pages on which OCR is close to perfect, so this result should not be extended to scans or photographs.
- **The kind of visual embedding matters far more than its presence (RQ2).** A single 224 px SigLIP2 vector per page is close to unusable for this task (0.094), while patch-level late interaction with a much smaller model reaches 0.608 (+0.514 [+0.469, +0.559]). OCR text from Tesseract matches the parser text shipped with the dataset (+0.002 [-0.007, +0.010]).
- **Hybrid retrieval helps within text and is neutral across modalities (RQ3).** Fusing BM25 and dense rankings improves on dense alone (+0.026 [+0.001, +0.050], p = 0.046, uncorrected). Adding ColSmol to the text hybrid changes nDCG@10 by +0.007 [-0.011, +0.024]. Adding SigLIP2 at equal weight lowers it by 0.133 [-0.164, -0.103], and still by 0.023 at a quarter of the weight: rank fusion does not protect against a weak component.
- **Reranking is not worth its cost on a strong first stage (RQ4).** A MiniLM cross-encoder over the top 20 changes the text hybrid by +0.001 [-0.024, +0.025] at about 20 times the latency (14 ms to 271 ms). It does repair a damaged ranking: the equal-weight SigLIP hybrid recovers from 0.556 to 0.691.
- **Complete evidence sets are rarely retrieved.** The best strategy places every relevant page in the top 10 for 43% of queries.
- **Human-written queries are easier than synthetic ones for lexical retrieval.** BM25 scores 0.676 on the 110 human-written queries and 0.589 on the 105 synthetic ones, so the synthetic queries do not favour word overlap with the page text.

### Generation

SmolVLM-500M (float16, Apple MPS) on 60 answerable questions sampled with a fixed seed, the same 60 with every relevant page withheld, and 30 questions written for a different corpus. Retrieval is the text hybrid; every evidence mode receives the same three pages.

| Evidence given to the model | Token F1 | ROUGE-L | Abstains, evidence withheld | Abstains, out of domain | Cites a page | Generation s |
|---|---|---|---|---|---|---|
| None (direct model) | 0.251 [0.198, 0.305] | 0.203 | n/a | 0.367 | n/a | 0.72 |
| OCR text of retrieved pages | 0.343 [0.285, 0.406] | 0.290 | 0.017 | 0.100 | 0.067 | 1.43 |
| Images of retrieved pages | 0.299 [0.239, 0.364] | 0.248 | 0.000 | 0.033 | 0.100 | 1.80 |
| Images and OCR text | 0.290 [0.233, 0.346] | 0.249 | 0.000 | 0.000 | 0.067 | 2.29 |
| Images of the annotated relevant pages | 0.306 [0.245, 0.369] | 0.260 | n/a | n/a | 0.085 | 1.64 |

Findings:

- **Retrieved text improves a small model; retrieved page images do not clearly do so.** Text RAG gains +0.093 token F1 over the direct model ([+0.032, +0.159], p = 0.003). Image RAG gains +0.049 ([-0.026, +0.124], p = 0.20). Text against image evidence is not separable at this sample size (-0.044 [-0.111, +0.020]).
- **A 0.5 B parameter model does not abstain when the evidence is missing (RQ5).** With every relevant page withheld it still answers 98 to 100% of the questions, and it answers 90 to 100% of out-of-domain questions when given retrieved pages. It abstains more often with no evidence at all (37% out of domain) than with irrelevant evidence: supplying pages makes it more willing to answer, whatever they contain.
- **The citation format is mostly ignored.** A page is cited in 7 to 10% of answers, so citation precision and recall rest on about five answers per system and are not interpretable for this model.
- **Better retrieval would not fix the generator.** Giving the model the annotated relevant pages instead of the retrieved ones changes token F1 by +0.006 [-0.035, +0.050].
- **Token log-probability is a weak grounding signal.** Its AUROC for separating answers produced with a relevant page in context from answers produced without one is 0.53 to 0.63.

Token F1 against full-sentence reference answers understates correctness, and the faithfulness metric cannot credit support that exists only in pixels; both limitations are described in [docs/evaluation.md](docs/evaluation.md). Outputs are deterministic: a repeated run reproduced all 240 direct and text-RAG answers exactly.

### Qwen3-VL-2B comparison (Colab T4)

The pinned 2B model was evaluated with the same questions and frozen retrieved
pages as SmolVLM: 540 generations, float16, greedy decoding, 160-token limit.
All output checksums and paired inputs were verified locally before CPU scoring.

| Evidence | SmolVLM token F1 | Qwen token F1 | Qwen minus SmolVLM [paired 95% interval] |
|---|---:|---:|---|
| None | 0.251 | 0.342 | +0.091 [+0.027, +0.149] |
| OCR text | 0.343 | 0.395 | +0.051 [-0.025, +0.128] |
| Images | 0.299 | 0.353 | +0.054 [-0.022, +0.126] |
| Images + OCR text | 0.290 | 0.390 | +0.101 [+0.035, +0.166] |

Intervals are unadjusted paired bootstrap intervals over 60 answerable questions.
Both model family and parameter count change, so these are not pure scaling effects.
Qwen cites pages on 38–49 of 60 answerable questions per RAG mode and abstains on
25–28 of 30 out-of-domain questions. Citation recall is 0.475–0.670, but the
NLI faithfulness proxy remains 0.347–0.372: more frequent citations do not imply
that every answer is supported. See the full [results](results/SUMMARY.md) and
[machine-readable paired comparison](results/analysis/qwen3vl_2b_vs_smolvlm_500m.json).

The full run took 52 minutes. Peak allocated VRAM was 7.30 GiB; peak reserved
VRAM reached 14.28 GiB, with one allocator warning followed by successful
completion. 115/540 outputs reached the token limit. Hardware timings are not
comparable to the MPS baseline. Recovery and provenance are documented in
[COLAB_HANDOFF.md](docs/COLAB_HANDOFF.md).

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
