"""Prompt construction and answer parsing for grounded generation."""

from __future__ import annotations

import re

from mmrag.generate.vlm import Content, image_part, text_part
from mmrag.types import Answer, Evidence

ABSTAIN = "INSUFFICIENT_EVIDENCE"

GROUNDED_RULES = (
    "Answer the question using ONLY the evidence pages above. "
    "Each page is labelled [P1], [P2], and so on.\n"
    f"If the evidence does not contain the answer, reply with exactly: {ABSTAIN}\n"
    "Otherwise reply in exactly this format:\n"
    "ANSWER: <short answer>\n"
    "CITATIONS: <labels of the pages that support the answer, for example P1, P2>"
)

CLOSED_BOOK_RULES = (
    "Answer the question from your own knowledge. No documents are provided.\n"
    f"If you do not know the answer, reply with exactly: {ABSTAIN}\n"
    "Otherwise reply in exactly this format:\n"
    "ANSWER: <short answer>"
)

# Fixed captioning instruction. It never sees any evaluation query.
CAPTION_PROMPT = (
    "Describe this document page for a search index. State its main topic and section "
    "titles, and describe any figure, chart, table, diagram or code listing and what it "
    "shows. Be specific and concise."
)

RECOGNITION_PROMPT = (
    "Describe what is visible in this image: the type of content, any text, and any "
    "figures, charts or tables. Report only what you can see."
)


def build_prompt(
    question: str,
    evidence: list[Evidence],
    mode: str,
    images: dict[str, object] | None = None,
) -> Content:
    """mode: closed_book | text | image | image_text. `images` maps label -> PIL image."""
    if mode == "closed_book" or not evidence:
        rules = CLOSED_BOOK_RULES if mode == "closed_book" else GROUNDED_RULES
        return [text_part(f"{rules}\n\nQuestion: {question}")]
    content: Content = []
    for ev in evidence:
        if mode in ("image", "image_text"):
            content.append(text_part(f"[{ev.label}]"))
            content.append(image_part(images[ev.label]))
            if mode == "image_text" and ev.text:
                content.append(text_part(f"Extracted text of {ev.label}:\n{ev.text}"))
        else:
            content.append(text_part(f"[{ev.label}]\n{ev.text or '(no text extracted)'}"))
    content.append(text_part(f"{GROUNDED_RULES}\n\nQuestion: {question}"))
    return content


_ABSTAIN_RE = re.compile(r"insufficient[_ ]evidence", re.I)
_LABEL_RE = re.compile(r"\bP\s?(\d+)\b")


def parse_answer(raw: str, evidence: list[Evidence]) -> Answer:
    """Parse the generator reply into answer text, abstention flag and citations."""
    text = raw.strip()
    by_label = {e.label: e.page_id for e in evidence}

    m = re.search(r"ANSWER\s*:\s*(.*?)(?:\n\s*CITATIONS?\s*:|\Z)", text, flags=re.S | re.I)
    answer_text = (
        m.group(1) if m else re.split(r"\n\s*CITATIONS?\s*:", text, flags=re.I)[0]
    ).strip()
    c = re.search(r"CITATIONS?\s*:\s*(.*)", text, flags=re.S | re.I)
    # Fall back to inline [P1]-style mentions when no CITATIONS line is present.
    cite_src = c.group(1) if c else " ".join(re.findall(r"\[(P\s?\d+)\]", text))

    labels: list[str] = []
    for num in _LABEL_RE.findall(cite_src):
        label = f"P{num}"
        if label not in labels:
            labels.append(label)
    valid = [x for x in labels if x in by_label]
    invalid = [x for x in labels if x not in by_label]

    # An explicit abstention, or an empty answer, counts as abstaining.
    abstained = bool(_ABSTAIN_RE.search(answer_text)) or not answer_text
    if abstained:
        answer_text, valid = "", []
    return Answer(
        raw=raw,
        text=answer_text,
        abstained=abstained,
        cited_labels=valid,
        cited_page_ids=[by_label[x] for x in valid],
        invalid_citations=invalid,
    )
