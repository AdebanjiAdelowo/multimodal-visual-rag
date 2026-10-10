# Research audit

This document states, for each research question, what the repository can and cannot currently support, and lists the threats to validity. Experiment status and commands are in [experiments.md](experiments.md).

## Claims and evidence

| RQ | Evidence available | What can be claimed |
|---|---|---|
| RQ1 image-text vs text-only retrieval on visual questions | Text strategies are evaluated with a visual / text-only split. Visual strategies are implemented; their evaluation at the current code revision is pending. | Text retrieval loses 0.03 to 0.09 nDCG@10 on visually grounded questions. No claim yet about whether visual retrieval closes that gap. |
| RQ2 OCR vs captions vs visual embeddings | OCR (two text sources) evaluated. Captions and visual embeddings pending. | OCR text from Tesseract is as good as the dataset's parser text for BM25 on this corpus. |
| RQ3 hybrid retrieval | Text hybrid evaluated. Multimodal hybrids pending. | BM25 + dense fusion improves over dense alone (+0.026 nDCG@10, p = 0.046, uncorrected). |
| RQ4 reranking | Evaluated for the text hybrid. | No measurable gain (+0.001 [-0.024, +0.025]) at 12 times the latency, with this reranker on this corpus. |
| RQ5 unsupported questions | Harness, conditions and metrics implemented and unit-tested. No model runs yet. | Nothing yet. |
| RQ6 local vs GPU trade-off | Timers and memory accounting implemented. Local retrieval latency measured under load. GPU runs pending. | Indicative local retrieval latency only. |

## Threats to validity

**Sample size.** 215 queries. The 95% interval on nDCG@10 is about +/- 0.035, and the visual strata have 119 and 96 queries. Per-type strata (chart: 10, image: 5) are too small for any conclusion.

**Single domain and clean pages.** Two textbooks from one publisher, rendered from digital sources. OCR is near perfect here; conclusions about OCR versus visual embeddings will not transfer to scans, photographs or handwriting.

**Multiple comparisons.** Paired tests are reported without correction. The one comparison at p = 0.046 would not survive a Bonferroni correction across the comparisons listed in the report.

**Relevance annotation.** Judgements are human-verified but not exhaustive. Unjudged relevant pages depress every strategy's score and can make the missing-evidence condition answerable.

**Synthetic queries.** 105 of 215 queries were produced by a generation pipeline before human verification. If that pipeline conditioned on parsed page text, lexical methods may be favoured on those queries. The `query_generator` field is preserved in the query metadata so results can be split by origin; this split is not yet reported.

**Automatic generation metrics.** See evaluation.md. In particular, the faithfulness metric cannot credit visual support, which will bias image-mode systems downward on visual questions relative to their true faithfulness.

**Contamination.** The benchmark and the OpenStax textbooks are public and may be in model pretraining data. The closed-book condition measures how much a model can answer without evidence and should be read with this in mind.

**Prompt sensitivity.** One prompt per mode, no tuning. Small models may fail the output format rather than the task; the parser reports format failures as answers without citations rather than as abstentions, and the has-citation rate is reported so the two can be told apart.

**Latency measurements.** Local timings so far were taken on a machine running other workloads. A clean re-run is required before RQ6 can be answered.

## Decisions that differ from the obvious default

- **ViDoRe v3 instead of DocVQA or ChartQA.** The latter have no corpus-level relevance judgements and more restrictive or copyleft terms. See datasets.md.
- **Page as the retrieval unit.** The relevance judgements are per page. Dense text indexes chunk internally and max-pool to pages.
- **Exact search, no vector database.** The corpus has 1,360 pages.
- **Same generator for text and image modes.** This isolates evidence modality from model quality. It also means text mode is served by a vision-language model, which may be weaker at long text than a text-only model of similar size.
- **Tesseract rather than a neural OCR model.** It is a widely available baseline and gives the same BM25 quality as the dataset's parser text on this corpus.
- **Lowercased SigLIP queries.** SigLIP text towers were trained on lowercased captions; lowercasing raised nDCG@10 from 0.087 to 0.094 in a development check on the full query set.

## Reproducibility

- Dataset revision is recorded in `meta.json` of the materialised corpus (`d5cc75883d92e294f0c0fc2662551c9708a06ebc` for the reported runs).
- Dependencies are pinned in `uv.lock`.
- Each result file embeds the full strategy configuration and a hardware summary.
- Sampling and bootstrap seeds are fixed. Decoding is greedy.
- Model revisions are not pinned; results may change if model repositories are updated.
