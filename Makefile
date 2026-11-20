PY := .venv/bin/python
DEMO ?=            # set DEMO=--demo to run on the synthetic set

.PHONY: setup model docker demo augment baseline transformer transformer-full distilbert export evaluate fertility zeroshot serve test mobile-lexicon

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
	cd mobile && npm install

model:                     ## download the trained int8 ONNX bundle from the GitHub release
	sh scripts/fetch_model.sh

docker:                    ## build and run the API container (needs artifacts/model)
	docker compose up --build -d

demo:                      ## full pipeline on synthetic data (smoke test, ~5 min on CPU)
	$(PY) scripts/make_demo_data.py
	$(MAKE) augment baseline evaluate DEMO=--demo
	$(PY) scripts/train_transformer.py --model afroxlmr --epochs 2 --demo
	$(PY) scripts/export_onnx.py --src artifacts/afroxlmr_torch --out artifacts/model --demo
	$(PY) scripts/evaluate.py --bundles artifacts/baseline artifacts/afroxlmr_torch artifacts/model --demo

augment:
	$(PY) scripts/augment.py $(DEMO)
baseline:
	$(PY) scripts/train_baseline.py $(DEMO)
transformer:               ## AfroXLM-R, Strathmore data only
	$(PY) scripts/train_transformer.py --model afroxlmr $(DEMO)
transformer-full:          ## + stage-1 intermediate fine-tuning on GoEmotions/AfriSenti/tabularisai
	$(PY) scripts/train_transformer.py --model afroxlmr --intermediate $(DEMO)
distilbert:                ## comparison model
	$(PY) scripts/train_transformer.py --model distilbert $(DEMO)
export:
	$(PY) scripts/export_onnx.py --src artifacts/afroxlmr_torch --out artifacts/model $(DEMO)
evaluate:
	$(PY) scripts/evaluate.py --bundles artifacts/baseline $(wildcard artifacts/*_torch) $(wildcard artifacts/model) $(DEMO)
fertility:
	$(PY) scripts/fertility.py $(DEMO)
zeroshot:
	$(PY) scripts/zero_shot.py $(DEMO)
mobile-lexicon:
	$(PY) scripts/export_mobile_lexicon.py
serve:                     ## dev server (draft library allowed)
	ALLOW_DRAFT_LIBRARY=1 COUNSELLOR_API_KEY=dev-key .venv/bin/uvicorn serving.app:app --reload --port 8000
test:
	$(PY) -m pytest -q
