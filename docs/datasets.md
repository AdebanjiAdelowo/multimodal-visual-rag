# Dataset review

Checked on 2026-10-09 against the Hugging Face dataset API and dataset cards. "Qrels" means page-level relevance judgements that say which pages are relevant to a query within a shared corpus.

## Requirements

1. Page images of visually rich documents (text, tables, charts, diagrams).
2. Page-level relevance judgements over a shared corpus, so retrieval can be scored without assumptions.
3. Reference answers, so generation can be scored on the same questions.
4. A licence that permits research use and, ideally, redistribution of derived metrics.
5. Small enough to index and evaluate on a laptop.

## Candidates

| Dataset | Licence (as declared) | Qrels | Answers | Size | Decision |
|---|---|---|---|---|---|
| ViDoRe v3, computer science (`vidore/vidore_v3_computer_science`) | CC-BY-4.0; source documents are OpenStax textbooks, CC-BY-4.0 | Graded (1, 2), human-verified, with bounding boxes; 4.9 relevant pages per query | Yes | 512 MB, 1,360 pages | **Selected** |
| ViDoRe v3, other domains (HR, finance, energy, pharmaceuticals, physics, industrial) | CC-BY-4.0 | Same schema | Yes | 0.4 to 2.7 GB each | Same loader; HR query text is used for out-of-domain probes |
| ViDoRe v2 economics reports | CC-BY-3.0 | Yes | Yes | 55 MB, 452 pages | Suitable, smaller and older than v3 |
| ViDoRe v2 biomedical lectures | CC-BY-NC-SA-4.0 | Yes | Yes | 78 MB | Non-commercial; not used |
| ViDoRe v2 ESG reports | none declared | Yes | Yes | 0.8 to 1.5 GB | No licence; not used |
| ViDoRe v1 DocVQA, InfoVQA subsets | MIT tag on the repackaging; underlying DocVQA/InfographicVQA data is shared for non-commercial research and education | One page per query (the page the question was written for) | Partly | 0.2 to 0.3 GB | Not used: source terms are more restrictive than the tag, and relevance is assumed rather than judged |
| ViDoRe v1 ArxivQA, TAT-DQA | CC-BY-SA-4.0, CC-BY-4.0 | One page per query | Yes | 77 to 136 MB | Candidates for a chart/figure extension |
| ChartQA (`HuggingFaceM4/ChartQA`) | GPL-3.0 | None | Yes | 1 GB | Not used: a VQA set with no corpus-level relevance; copyleft |
| VisRAG retrieval sets built on ChartQA and SlideVQA | none declared | Yes (derived) | No | 24 to 104 MB | No licence; not used |
| SlideVQA (`NTT-hil-insight/SlideVQA`) | none declared; gated | Evidence slides within a deck | Yes | 11 GB | Not used: gated, unlicensed, too large |
| MMDocIR evaluation set | Apache-2.0 | Page and layout level | Yes | 11 GB | Good fit, too large for the local disk |
| MMLongBench-Doc | none declared | Evidence pages within one document | Yes | 0.7 GB | No licence; not used |

## Why a VQA dataset is not a retrieval benchmark

DocVQA and ChartQA pair each question with the single image it was written about. Treating that image as the only relevant item in a pooled corpus gives a usable retrieval task, and ViDoRe v1 does this, but it has two weaknesses: questions written while looking at one page are often underspecified out of context ("what is the total?"), and other pages that also answer the question are scored as errors. ViDoRe v3 addresses both: queries are written or generated without the page in view and relevance is verified by annotators across the corpus, which is why queries have several relevant pages each.

## Properties of the selected corpus

- 1,360 pages: Introduction to Computer Science (945) and Introduction to Python Programming (415), OpenStax, 2024.
- 215 English queries after filtering the multilingual set: 110 human-written, 105 produced by a synthetic pipeline and verified by annotators. Formats: 115 questions, 65 instructions, 35 keyword queries.
- 1,049 relevance judgements (877 with grade 1, 172 with grade 2).
- Evidence modality labels per query: text 209, table 77, infographic 47, chart 10, image 5. 119 queries have at least one non-text label.
- The dataset ships a `markdown` field with parsed page text. It is stored as the text source `parser_markdown` and used only as a reference parser; the OCR baselines use text this project extracts from the pixels with Tesseract 5.5.

## Unanswerable questions

The benchmark has no unanswerable questions, so two transparent constructions are used:

- **Evidence withheld.** For an answerable question, every page with a positive relevance grade is removed from the candidate pool before ranking. Limitation: relevance annotation may be incomplete, so a withheld-evidence question can occasionally still be answerable from an unlabelled page. This biases measured abstention downward.
- **Out of domain.** Query text from the ViDoRe v3 HR domain (European Union human-resources reports) is asked against the computer science corpus. Only the query text is downloaded and used. A few of these could in principle overlap with textbook content; they are not manually filtered.

## Leakage controls

- The page manifest holds page identifier, document identifier, page number and image path only.
- Queries, answers and relevance labels live in `queries.jsonl` and `qrels.jsonl`, which no indexing, OCR or captioning module opens.
- The captioning prompt is a fixed string; captions depend only on the page image.
- The benchmark is public, so the pages or questions may appear in the pretraining data of any model used here. This cannot be ruled out and applies equally to all published results on the benchmark.

## Data handling

Nothing downloaded is committed. `data/`, `artifacts/` and `uploads/` are Git-ignored. Committed result files contain metric values, strategy configurations and hardware descriptions.
