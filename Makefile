VENV = .venv
PY   = $(VENV)/bin/python
PIP  = $(VENV)/bin/pip
IMAGE = touchpoint-lab:latest
RAW_EVENTS = data/raw_events.csv

setup: $(VENV)/bin/activate
$(VENV)/bin/activate: requirements.txt
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $@

test:
	$(VENV)/bin/pytest -q

demo-taxonomy:
	$(PY) -m src.event_taxonomy \
		--csv $(RAW_EVENTS) \
		--out data/value_features.parquet

demo-guardrail:
	$(PY) -m src.guardrail_model train \
		--csv $(RAW_EVENTS) \
		--model models/guardrail.pkl && \
	$(PY) -m src.guardrail_model predict \
		--model models/guardrail.pkl \
		--n_events_24h 4 --pct_brand_action 0.05

notebook:
	$(VENV)/bin/jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --port=8888

clean:
	rm -rf $(VENV) .pytest_cache **/*.pkl data/*.parquet

docker-build:
	docker build -t $(IMAGE) .

docker-test: docker-build
	docker run --rm $(IMAGE)

docker-bash: docker-build
	docker run --rm -it $(IMAGE) /bin/bash

.PHONY: setup test demo-taxonomy demo-guardrail notebook clean docker-build docker-test docker-bash
