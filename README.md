# touchpoint-lab 🧪  
Rapid-prototype workspace for **event taxonomy, predictive value scoring, and engagement-risk guardrails** on customer touch-point data.

---

## 📂 Repo Layout
```
├── src/                      # production-style modules
│   ├── event_taxonomy.py
│   ├── action_classifier.py
│   └── guardrail_model.py
├── touchpoint-lab/
│   └── data/
│       ├── raw_events.csv    # 20 k synthetic Klaviyo-style events
│       └── labeled_event_actions.csv
├── notebooks/                # exploratory analyses
│   ├── 01_taxonomy_value.ipynb
│   └── 02_guardrail.ipynb
├── tests/                    # pytest suite
├── requirements.txt
├── Makefile                  # dev shortcuts (setup, tests, demo, docker…)
├── .pre-commit-config.yaml   # black • isort • ruff • nbstripout
└── Dockerfile
```

---

## 🚀 Quick Start (local)

```bash
# 1. create & activate env
conda create -n touchpoint-lab python=3.11
conda activate touchpoint-lab
pip install -r requirements.txt

# 2. run unit tests
make test

# 3. demo the two CLI pipelines
make demo-taxonomy     # clusters events + prints SHAP ranking
make demo-guardrail    # trains survival RSF + prints cooldown rec
```

Open notebooks:

```bash
make notebook          # launches JupyterLab on http://localhost:8888
```

---

## 🐳 100 % Repro in Docker

```bash
make docker-build      # builds image (tag: touchpoint-lab:latest)
make docker-test       # runs pytest inside the container
```

---

## ✨ Key Features

| Module | What it does | CLI |
|--------|--------------|-----|
| **event_taxonomy.py** | FastText + HDBSCAN clusters noisy `event_name`s → canonical IDs; LightGBM computes mean \|SHAP\| lift on 7-day purchase | `python -m src.event_taxonomy --csv … --out …` |
| **action_classifier.py** | Few-shot TF‑IDF + log‑reg classifier → `CUSTOMER` vs `BRAND` actions; used as a feature | `python -m src.action_classifier train|predict …` |
| **guardrail_model.py** | Random‑Survival‑Forest predicts 7‑day unsubscribe/spam risk; converts to 0‑30‑day cool‑down window | `python -m src.guardrail_model train|predict …` |

---

## 🔬 Data Schema (`raw_events.csv`)

| column | dtype | example |
|--------|-------|---------|
| `profile_id` | int | `1553` |
| `event_ts`   | ISO datetime | `2025-02-14T09:22:17` |
| `event_name` | str | `Clicked email link` |
| `purchased_7d` | int (0/1) | `1` |

---

## 🧹  Code Quality

```bash
# install git hooks once
make precommit-install

# auto-format & lint on demand
make fmt
make lint
```

Pre‑commit runs **Black**, **isort**, **Ruff**, and **nbstripout** on every commit.

---

## 📝  Roadmap (beyond MVP)

* Vertex AI Feature Store & Pipelines for daily retraining  
* Contextual‑bandit SMS optimiser (T‑Learner + ε‑greedy)  
* Isolation‑Forest bot detector integrated into guardrail  
* Vector embeddings (Matching Engine) for launch‑audience retrieval  
* Identity‑resolution graph with human‑in‑loop review
