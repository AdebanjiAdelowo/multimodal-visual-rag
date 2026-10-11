"""Answer-quality, grounding and citation metrics.

Lexical overlap (token F1, ROUGE-L) follows the SQuAD / ROUGE definitions. The
entailment-based scores use a small NLI cross-encoder and are automatic
proxies: they are reported alongside, never instead of, the lexical scores.
See docs/evaluation.md for the exact protocol and its known failure modes.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from mmrag.text import chunk_words

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation, articles and extra spaces."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Share of reference tokens present in the prediction."""
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not r:
        return float(not p)
    return sum((Counter(p) & Counter(r)).values()) / len(r)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 on normalised tokens."""
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def polarity(text: str) -> str | None:
    """'yes' / 'no' if the text opens with one, else None."""
    m = re.match(r"\W*(yes|no)\b", text.strip().lower())
    return m.group(1) if m else None


def boolean_accuracy(prediction: str, reference: str) -> float | None:
    """Yes/no agreement; None when the reference is not a yes/no answer."""
    ref = polarity(reference)
    if ref is None:
        return None
    return float(polarity(prediction) == ref)


def citation_scores(
    cited: list[str], gold: set[str], provided: list[str]
) -> dict[str, float | None]:
    """Citation precision and recall against the annotated relevant pages.

    precision: share of cited pages that are annotated relevant.
    recall: share of the relevant pages shown to the generator that it cited.
    Both are None when undefined (nothing cited / no relevant page was shown).
    """
    cited_set = set(cited)
    reachable = gold & set(provided)
    return {
        "citation_precision": len(cited_set & gold) / len(cited_set) if cited_set else None,
        "citation_recall": len(cited_set & reachable) / len(reachable) if reachable else None,
    }


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


class NLIScorer:
    """Entailment probabilities from an NLI cross-encoder."""

    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-xsmall", device: str = "auto"):
        from sentence_transformers import CrossEncoder

        if device == "auto":
            # CUDA when present; otherwise CPU, which leaves MPS memory to the generator.
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model
        self.model = CrossEncoder(model, device=device)
        id2label = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self.entail_idx = next(i for i, v in id2label.items() if v.startswith("entail"))

    def entail(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        if not pairs:
            return []
        import numpy as np

        logits = np.asarray(self.model.predict(pairs, batch_size=32, show_progress_bar=False))
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs[:, self.entail_idx].tolist()

    def faithfulness(
        self, answer: str, evidence_texts: list[str], window: int = 90
    ) -> float | None:
        """Mean over answer sentences of the best entailment by any evidence window.

        Evidence text is split into overlapping word windows because NLI models
        are trained on short premises. Returns None if there is no answer or no
        evidence text. Evidence that exists only as pixels (a chart with no
        transcribable text) cannot support a claim under this metric.
        """
        sentences = split_sentences(answer) or ([answer.strip()] if answer.strip() else [])
        windows = [
            w
            for t in evidence_texts
            if t and t.strip()
            for w in chunk_words(t, window, window // 3)
        ]
        if not sentences or not windows:
            return None
        scores = []
        for s in sentences:
            scores.append(max(self.entail([(w, s) for w in windows])))
        return float(sum(scores) / len(scores))

    def answer_entailment(self, question: str, prediction: str, reference: str) -> float:
        """P(reference answer entails the predicted answer), both prefixed by the question.

        A precision-flavoured correctness proxy: high when the prediction says
        nothing the reference does not support.
        """
        return self.entail([(f"{question} {reference}", f"{question} {prediction}")])[0]
