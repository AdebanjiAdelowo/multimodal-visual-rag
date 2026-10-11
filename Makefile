CORPUS ?= data/processed/vidore_v3_computer_science
RETRIEVAL ?= configs/retrieval/*.yaml

.PHONY: install test lint data ocr retrieval report api ui kaggle-bundle kaggle-kernel kaggle-submit kaggle-status

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

# Upload folders for the Kaggle API, built from the committed tree only:
#   kaggle/build/dataset  source archive + the commit it was made from
#   kaggle/build/kernel   the runner script and its kernel metadata
kaggle-kernel:
	@git diff --quiet HEAD || (echo "commit or stash changes first: the bundle is built from HEAD" && exit 1)
	rm -rf kaggle/build && mkdir -p kaggle/build/dataset kaggle/build/kernel
	git archive --format=zip --prefix=multimodal-visual-rag/ -o kaggle/build/dataset/mmrag-src.zip HEAD
	git rev-parse HEAD > kaggle/build/dataset/COMMIT
	cp kaggle/dataset-metadata.json kaggle/build/dataset/
	git show HEAD:kaggle/run_gpu.py > kaggle/build/kernel/run_gpu.py
	cp kaggle/kernel-metadata.json kaggle/build/kernel/

# The Kaggle client is not a project dependency; point KAGGLE_PYTHON at an interpreter that has it.
KAGGLE_PYTHON ?= python3
KAGGLE = $(KAGGLE_PYTHON) kaggle/kaggle_ipv4.py
KERNEL = adebanjiadelowo/multimodal-visual-rag-gpu

# Start the batch job from the folder built by `make kaggle-kernel` (uses GPU quota).
kaggle-submit:
	$(KAGGLE) kernels push -p kaggle/build/kernel --accelerator NvidiaTeslaT4

kaggle-status:
	$(KAGGLE) kernels status $(KERNEL)
