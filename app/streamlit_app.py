"""Streamlit front end for the Multimodal Visual RAG API.

    uvicorn mmrag.api.app:app --port 8000
    streamlit run app/streamlit_app.py

The UI keeps three things apart, each in its own tab:
  Search      what the retriever found (no model-generated text)
  Ask         an answer grounded in retrieved pages, with page citations
  Recognise   what a model sees in one uploaded image (no corpus, no citations)
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.environ.get("MMRAG_API_URL", "http://localhost:8000")
MODES = {
    "image": "Page images",
    "image_text": "Page images + OCR text",
    "text": "OCR text only",
    "closed_book": "No retrieval (model knowledge only)",
}

st.set_page_config(page_title="Multimodal Visual RAG", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the API; show the server's error message instead of a stack trace."""
    try:
        r = httpx.request(method, f"{API}{path}", timeout=600, **kwargs)
    except httpx.HTTPError as e:
        st.error(f"Cannot reach the API at {API}: {e}")
        st.stop()
    if r.status_code >= 400:
        detail = (
            r.json().get("detail", r.text)
            if "json" in r.headers.get("content-type", "")
            else r.text
        )
        st.error(f"{r.status_code}: {detail}")
        return None
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def page_image(url: str) -> bytes:
    return httpx.get(f"{API}{url}", timeout=60).content


def page_card(hit: dict, label: str | None = None, cited: bool = False) -> None:
    title = f"{label + ' · ' if label else ''}{hit['doc_id']} · page {hit['page_number'] + 1}"
    st.markdown(f"**{title}**" + (" · cited" if cited else ""))
    st.image(page_image(hit["image_url"]), use_container_width=True)
    st.caption(f"rank {hit['rank']} · score {hit['score']:.4f}")
    if hit.get("components"):
        with st.expander("Score components"):
            st.json(hit["components"])


def page_grid(hits: list[dict], labels: dict[str, str] | None = None, cited: set | None = None):
    cols = st.columns(3)
    for i, h in enumerate(hits):
        with cols[i % 3]:
            page_card(h, (labels or {}).get(h["page_id"]), h["page_id"] in (cited or set()))


# ------------------------------------------------------------------ sidebar
health = api("GET", "/health")
corpora = api("GET", "/corpora") or []
strategies = api("GET", "/strategies") or {"available": [], "default": None}

with st.sidebar:
    st.header("Corpus")
    if not corpora:
        st.info("No corpus available yet. Upload a document below.")
    names = [c["name"] for c in corpora]
    corpus = st.selectbox("Corpus", names) if names else None
    if corpus:
        info = next(c for c in corpora if c["name"] == corpus)
        st.caption(
            f"{info['n_pages']} pages in {info['n_documents']} documents"
            + (f" · {info['license']}" if info.get("license") else "")
        )
    default = strategies["default"]
    avail = strategies["available"]
    strategy = st.selectbox(
        "Retrieval strategy", avail, index=avail.index(default) if default in avail else 0
    )
    st.caption(f"Generator: {health['generator'] if health else 'unknown'}")

    st.header("Upload")
    files = st.file_uploader(
        "PDF or image", type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True
    )
    if files and st.button("Add to upload corpus"):
        with st.spinner("Parsing and running OCR"):
            res = api("POST", "/upload", files=[("files", (f.name, f.getvalue())) for f in files])
        if res:
            st.success(f"Added {len(res['pages_added'])} pages to '{res['corpus']}'")
            st.rerun()

st.title("Multimodal Visual RAG")
tab_ask, tab_search, tab_recognise = st.tabs(["Ask", "Search", "Recognise an image"])

# ---------------------------------------------------------------------- ask
with tab_ask:
    st.caption("Answers are generated only from the retrieved pages shown below.")
    question = st.text_input("Question", key="ask_q")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Evidence given to the model", list(MODES), format_func=MODES.get)
    max_pages = c2.slider("Pages given to the model", 1, 6, 3)
    if st.button("Ask", type="primary") and question and corpus:
        with st.spinner("Retrieving and generating"):
            res = api(
                "POST",
                "/ask",
                json={
                    "question": question,
                    "corpus": corpus,
                    "strategy": strategy,
                    "mode": mode,
                    "max_pages": max_pages,
                },
            )
        if res:
            st.subheader("Grounded answer")
            if res["abstained"]:
                st.warning("The model found insufficient evidence to answer.")
            else:
                st.write(res["answer"])
                if res["citations"]:
                    cites = ", ".join(
                        f"{c['label']} ({c['doc_id']} p.{c['page_number'] + 1})"
                        for c in res["citations"]
                    )
                    st.markdown(f"**Cited pages:** {cites}")
                elif res["mode"] != "closed_book":
                    st.warning("The model gave an answer without citing any page.")
                if res["invalid_citations"]:
                    st.error(f"Cited labels that were not provided: {res['invalid_citations']}")

            with st.expander("Uncertainty indicators (uncalibrated)"):
                u = res["uncertainty"]
                m1, m2, m3 = st.columns(3)
                lp = u["mean_token_logprob"]
                m1.metric("Mean token log-prob", f"{lp:.2f}" if lp is not None else "n/a")
                mg = u["retrieval_top1_margin"]
                m2.metric("Retrieval top-1 margin", f"{mg:.4f}" if mg is not None else "n/a")
                m3.metric("Valid citation", "yes" if u["has_valid_citation"] else "no")
                st.caption(
                    "These are raw signals, not probabilities of correctness. "
                    "See docs/evaluation.md for how well they track answer quality."
                )

            st.subheader("Evidence shown to the model")
            if res["evidence"]:
                labels = {e["page_id"]: e["label"] for e in res["evidence"]}
                cited = {c["page_id"] for c in res["citations"]}
                page_grid(res["evidence"], labels, cited)
            else:
                st.caption("None. The model answered without documents.")

            with st.expander("Timings, tokens and raw model output"):
                st.json({"timings_s": res["timings_s"], "tokens": res["tokens"]})
                st.code(res["raw_output"])

# ------------------------------------------------------------------- search
with tab_search:
    st.caption("Retrieval only. Nothing on this tab is generated by a language model.")
    kind = st.radio("Query by", ["Text", "Image"], horizontal=True)
    k = st.slider("Results", 3, 30, 9)
    if kind == "Text":
        q = st.text_input("Search query", key="search_q")
        if st.button("Search") and q and corpus:
            res = api(
                "POST", "/search", json={"query": q, "corpus": corpus, "strategy": strategy, "k": k}
            )
            if res:
                st.caption(f"{res['strategy']} · {res['latency_ms']:.0f} ms")
                page_grid(res["hits"])
    else:
        img = st.file_uploader("Query image", type=["png", "jpg", "jpeg", "webp"], key="qimg")
        st.caption("Uses the visual index (strategy visual_siglip), whatever is selected above.")
        if img and st.button("Find similar pages") and corpus:
            res = api(
                "POST",
                "/search/image",
                files={"file": (img.name, img.getvalue())},
                data={"corpus": corpus, "k": str(k)},
            )
            if res:
                page_grid(res["hits"])

# ---------------------------------------------------------------- recognise
with tab_recognise:
    st.caption(
        "Visual recognition only: what the model and OCR read from this one image. "
        "No corpus is consulted and nothing here is a cited, grounded claim."
    )
    img = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], key="rimg")
    if img:
        left, right = st.columns(2)
        left.image(img.getvalue(), use_container_width=True)
        if right.button("Recognise"):
            with st.spinner("Running the vision-language model"):
                res = api("POST", "/recognize", files={"file": (img.name, img.getvalue())})
            if res:
                right.markdown("**Model description**")
                right.write(res["description"])
                right.markdown("**OCR text**")
                right.code(res["ocr_text"] or "(none)")
