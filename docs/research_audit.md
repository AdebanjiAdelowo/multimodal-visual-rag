# Research audit

This document states, for each research question, what the repository can and cannot currently support, and lists the threats to validity. Experiment status and commands are in [experiments.md](experiments.md).

## Claims and evidence

| RQ | Evidence available | What can be claimed |
|---|---|---|
| RQ1 image-text vs text-only retrieval on visual questions | Text, single-vector visual and late-interaction visual retrieval evaluated on the full corpus with a visual / text-only split. ColQwen2 pending (GPU). | On this corpus, visual retrieval does not beat OCR-based retrieval on visually grounded questions: ColSmol-256M 0.572 vs dense OCR 0.630 nDCG@10 (-0.057 [-0.111, -0.010]). The claim is limited to small visual retrievers and born-digital pages. |
| RQ2 OCR vs captions vs visual embeddings | OCR (two text sources), SigLIP2 and ColSmol evaluated. Captions pending (GPU). | Tesseract text equals parser text for BM25. One 224 px vector per page is far below patch-level late interaction (0.094 vs 0.608). Nothing yet about captions. |
| RQ3 hybrid retrieval | Text hybrid and three multimodal hybrids evaluated; answer correctness measured for one generator with text-hybrid retrieval only. | BM25 + dense improves on dense (+0.026, p = 0.046 uncorrected). Adding ColSmol is neutral (+0.007 [-0.011, +0.024]); adding SigLIP2 is harmful. No claim yet that hybrid retrieval changes answer correctness: generation has not been run with a second retrieval strategy. |
| RQ4 reranking | Evaluated on the text hybrid, the SigLIP hybrid and the ColSmol hybrid. | No gain on a strong first stage (+0.001 and -0.001) at about 20 times the latency; a large gain (+0.135) on a first stage damaged by a weak component. |
| RQ5 unsupported questions | SmolVLM-500M under three evidence conditions (60 / 60 / 30 examples). Larger models pending (GPU). | A 0.5 B model abstains on 0 to 2% of questions whose evidence was withheld and cites a page in at most 10% of answers. Whether larger models behave differently is open. |
| RQ6 local vs GPU trade-off | Local latency and memory for all retrieval strategies (one session) and for SmolVLM-500M generation. GPU runs pending. | Local figures only. No comparison yet. |

## Threats to validity

**Sample size.** 215 queries. The 95% interval on nDCG@10 is about +/- 0.035, and the visual strata have 119 and 96 queries. Per-type strata (chart: 10, image: 5) are too small for any conclusion.

**Single domain and clean pages.** Two textbooks from one publisher, rendered from digital sources. OCR is near perfect here; conclusions about OCR versus visual embeddings will not transfer to scans, photographs or handwriting.

**Multiple comparisons.** Paired tests are reported without correction. The one comparison at p = 0.046 would not survive a Bonferroni correction across the comparisons listed in the report.

**Relevance annotation.** Judgements are human-verified but not exhaustive. Unjudged relevant pages depress every strategy's score and can make the missing-evidence condition answerable.

**Synthetic queries.** 105 of 215 queries were produced by a generation pipeline before human verification. If that pipeline conditioned on parsed page text, lexical methods could be favoured on those queries. The results are therefore split by query origin: BM25 scores lower on synthetic queries (0.589) than on human-written ones (0.676), so there is no sign of such an advantage. The two groups may differ in difficulty for other reasons, and the split is descriptive.

**Automatic generation metrics.** See evaluation.md. In particular, the faithfulness metric cannot credit visual support, which will bias image-mode systems downward on visual questions relative to their true faithfulness.

**Contamination.** The benchmark and the OpenStax textbooks are public and may be in model pretraining data. The closed-book condition measures how much a model can answer without evidence and should be read with this in mind.

**Prompt sensitivity.** One prompt per mode, no tuning. Small models may fail the output format rather than the task; the parser reports format failures as answers without citations rather than as abstentions, and the has-citation rate is reported so the two can be told apart.

**Latency measurements.** The reported local retrieval latencies come from one process on one session (load average about 6). They are single-run means; the same strategies measured a day earlier under heavier load were up to three times slower, so differences below that factor should not be interpreted. Generation latency for the first attempt of text RAG was inflated by memory swapping and was replaced by a repeat run with identical outputs.

**Small generation sample.** 60 answerable questions give intervals of about +/- 0.06 on token F1. Differences between evidence modes smaller than that are not resolved. Citation metrics for SmolVLM-500M rest on four to six answers per system.

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
