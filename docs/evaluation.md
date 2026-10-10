# Evaluation protocol

## Retrieval

For each query the pipeline ranks all pages; the top 20 are stored. With `rel(p)` the annotated grade of page `p` (0 if unjudged):

- **Recall@K**: relevant pages in the top K divided by all relevant pages. This is page-level evidence recall.
- **Hit@K**: 1 if any relevant page is in the top K.
- **Full Recall@K**: 1 if every relevant page is in the top K.
- **MRR@10**: reciprocal rank of the first relevant page within the top 10.
- **nDCG@K**: DCG with linear gain `rel(p)` and discount `1 / log2(rank + 1)`, normalised by the ideal ordering. This is trec_eval's `ndcg_cut`, the definition behind the ViDoRe leaderboards (computed there with pytrec_eval).

Aggregates are means over queries with 95% percentile-bootstrap intervals (2,000 resamples). Two strategies are compared with a paired bootstrap of the per-query difference (5,000 resamples); the reported p-value is two-sided. No correction for multiple comparisons is applied, so p-values near 0.05 should be read as suggestive.

Strata: `visual` (any gold evidence labelled table, chart, infographic or image), `text_only`, and one stratum per visual type.

Latency is wall-clock per query after one warm-up query, with per-stage timers for each index, fusion and reranking.

## Generation

### Conditions

| Condition | Evidence given to the generator | Expected behaviour |
|---|---|---|
| answerable | top pages from retrieval | answer, cite |
| missing_evidence | top pages after removing every relevant page from the pool | abstain |
| out_of_domain | top pages for a question about a different corpus | abstain |

Rankings are computed once per question and condition and reused by every generation system. An additional oracle variant gives the generator the annotated relevant pages (highest grade first), which bounds the generator independently of retrieval.

### Systems compared

| Evidence mode | What the model receives |
|---|---|
| closed_book | the question only (direct model, no retrieval) |
| text | OCR text of the selected pages |
| image | images of the selected pages |
| image_text | images and OCR text |

The same model serves all four modes.

### Output format and parsing

The prompt asks for `ANSWER: ...` and `CITATIONS: P1, P2`, or the single token `INSUFFICIENT_EVIDENCE`. Parsing is tolerant: a reply without the markers is taken as the answer with no citations; inline `[P2]` mentions are accepted as citations; an explicit abstention or an empty answer counts as abstaining; labels that were not supplied are recorded as invalid citations and excluded.

### Metrics

| Metric | Definition | Known weakness |
|---|---|---|
| Token F1 | SQuAD-normalised token overlap with the reference answer | Reference answers are full sentences; a correct short answer scores low |
| ROUGE-L | F1 of the longest common subsequence | Same |
| Yes/no accuracy | Agreement of leading yes/no, only where the reference starts with one | Small subset (20 boolean queries) |
| NLI entailment | P(reference entails prediction), question prepended to both, `nli-deberta-v3-xsmall` | Small NLI model; precision-flavoured; not validated against human grades |
| Faithfulness | Mean over answer sentences of the maximum entailment by any 90-word window of the supplied pages' text | Text only: a claim supported by a figure is counted as unsupported |
| Unsupported rate | Share of non-abstained answers with faithfulness below 0.5 | Inherits the above |
| Citation precision | Cited pages that are annotated relevant, over cited pages | Unjudged but relevant pages count against the model |
| Citation recall | Cited relevant pages over relevant pages that were shown | |
| Abstention rate | Share of abstentions, per condition | |
| Unsupported-answer rate (missing evidence, out of domain) | 1 minus abstention rate | Upper bound: see the note on incomplete annotation in datasets.md |

Abstentions score 0 on all correctness metrics in the answerable condition. Correctness is also reported on the subset where a relevant page was in the context, which separates generator errors from retrieval misses.

Token F1 and ROUGE-L are the primary correctness numbers because they are standard and deterministic. The published ViDoRe v3 end-to-end evaluation uses an LLM judge; that protocol is not reproduced here because no validated open judge was available on the target hardware. Scores are therefore not comparable with the paper's generation numbers. Retrieval nDCG@10 is comparable in definition, on the English subset.

### Confidence

The mean log-probability of the generated tokens is recorded for each answer. Its usefulness is tested by AUROC on a label that needs no human judgement: among non-abstained answers pooled from the answerable and missing-evidence conditions, does the score separate answers produced with a relevant page in context from answers produced without one? An AUROC near 0.5 means the signal carries no information about grounding and should not be shown as confidence.

### Cost

Per system: generation latency, input and output tokens, parameter count, weight footprint, peak resident memory and accelerator memory. No paid API is used, so token counts are reported as a proxy for cost.

## Sampling

Generation experiments on local hardware use a seeded sample of the 215 queries, stratified to preserve the visual to text-only ratio (`sample_queries`, seed 13). The sample size is recorded in every result file.
