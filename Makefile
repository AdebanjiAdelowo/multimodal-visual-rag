CORPUS ?= data/processed/vidore_v3_computer_science
RETRIEVAL ?= configs/retrieval/*.yaml

.PHONY: install test lint data ocr retrieval report api ui kaggle-bundle

install:
	uv venv --python 3.12 && uv pip install -e ".[models,app,dev]"

test:
	HF_HUB_OFFLINE=1 .venv/bin/pytest

lint:
	.venv/bin/ruff check src tests app kaggle && .venv/bin/ruff format --check src tests app kaggle

data:
	.venv/bin/mmrag fetch-vidore --domain computer_science --drop-raw
	.venv/bin/mmrag fetch-ood -c $(CORPUS)

ocr:
	.venv/bin/mmrag ocr -c $(CORPUS)

retrieval:
	.venv/bin/mmrag eval-retrieval $(RETRIEVAL) -c $(CORPUS)

report:
	.venv/bin/mmrag report

api:
	.venv/bin/uvicorn mmrag.api.app:app --port 8000

ui:
	.venv/bin/streamlit run app/streamlit_app.py

# Source archive of the current commit, to upload as a Kaggle dataset.
kaggle-bundle:
	git archive --format=zip --prefix=multimodal-visual-rag/ -o kaggle/mmrag-src.zip HEAD
