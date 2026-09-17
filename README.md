# Model Extraction Lab — Stealing ML Models via Prediction APIs

Reproduction of Tramèr et al. USENIX Security 2016. Victim API serves secret models. Attacker steals them with black-box queries only.

## Quickstart

Backend:

```bash
python3 -m pip install -r backend/requirements.txt
uvicorn app:app --port 8000 --app-dir backend
```

Frontend:

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Open the Vite URL, pick a victim, press Run extraction.

## What is implemented

- Binary LR equation solving with d plus 1 queries, exact solve on logit
- Softmax and MLP soft-label retraining with fidelity curve
- Decision tree path finding lite with leaf id plus line search
- Label-only baseline and defenses: full, rounded, label-only, noisy
- Custom synthetic fraud and triage datasets, fixed seeds

## API

- GET /models
- POST /predict with model_id, x, defense
- POST /attack with model_id, defense, budget
- GET /tree/{model_id}
