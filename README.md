# Model Extraction Lab

A reproduction of the model extraction attacks from "Stealing Machine Learning Models via Prediction APIs" by Tramèr, Zhang, Juels, Reiter, and Ristenpart, published at USENIX Security 2016.

A FastAPI service hosts four victim models behind a prediction API. A React dashboard runs a black-box attacker against them and reports how much of the victim the attacker recovered.

## How The Attacks Work

The paper starts from a simple observation. A prediction API gives away more than a class label. When the API returns a confidence score, a binary logistic regression turns into a system of linear equations. Query d plus 1 random points and the attacker solves for the weights exactly. A multiclass model needs c times d plus 1 queries, where c is the number of classes.

Decision trees leak a different kind of information. A confidence score records how training points fell along the path to a leaf, so two inputs on the same leaf return the same score. The attacker treats the score as a leaf identifier, searches each feature for the thresholds where it changes, and rebuilds the tree.

When the API hides confidence and returns only labels, the attacker falls back to retraining. It pulls labels for a batch of random points and trains a substitute model on them. This costs far more queries than equation solving and still recovers most of the victim's behavior.

## What Is Implemented

Four victims train on synthetic data with fixed seeds. The datasets are generated in memory at startup, so every run reproduces the numbers below.

| ID | Model | Data |
|---|---|---|
| `fraud_lr` | Binary logistic regression | 8 fraud features |
| `multi_lr` | Multinomial softmax | 6 triage features, 3 classes |
| `mlp` | Neural network, one hidden layer | 6 triage features, 3 classes |
| `tree` | Decision tree, depth 5 | 8 fraud features |

Four defenses change what the API returns.

| Defense | Response |
|---|---|
| `full` | Exact confidence scores |
| `rounded` | Confidence scores rounded to two decimals |
| `label_only` | The class label only |
| `noisy` | Confidence scores with added Gaussian noise |

Four attacks run against them. Equation solving for the binary model, soft label retraining for the softmax and the network, path finding for the tree, and a label-only retraining baseline.

## Setup

Install Python 3.9 or newer and Node.js 18 or newer.

Clone the repository.

```bash
git clone https://github.com/YashDahiwlikar914/Stealing-ML-Models-Via-Prediction-APIs.git
cd Stealing-ML-Models-Via-Prediction-APIs
```

Start the backend.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app:app --port 8000 --app-dir backend
```

The `--app-dir backend` flag is required. The backend modules import each other by plain name, so uvicorn has to run with `backend` on the module path.

Start the frontend in a second terminal.

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Open http://localhost:5173. The Vite dev server proxies `/models`, `/predict`, `/attack`, and `/tree` to port 8000, so the backend has to be running first.

## Using The Lab

Pick a victim, pick a defense, set a query budget, and press Run Extraction. The Lab tab shows the attack used, the fidelity against a held-out test set, the queries spent, and a fidelity curve for the retraining attacks. The Tree tab prints the victim tree.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/models` | none | Victim list, descriptions, test accuracy, defenses |
| POST | `/predict` | `model_id`, `x`, `defense` | Label and confidence scores |
| POST | `/attack` | `model_id`, `defense`, `budget` | Attack type, queries, fidelity, details |
| GET | `/tree/{model_id}` | none | Nested tree node structure |

## Verification

The end-to-end script runs every attack and prints fidelity numbers.

```bash
python backend/testE2E.py
```

Expected output.

```text
binary_lr fidelity: 0.9994444444444445 queries: 9
softmax fidelity: 0.9938888888888889
tree fidelity: 0.9572222222222222 leaves: 25
```

## Results

On the held-out test sets, the binary logistic regression is stolen exactly in 9 queries at a fidelity of 1.0000. The softmax reaches 0.9939 and the network 0.9828 with soft label retraining over 800 queries. Path finding recovers the tree at 0.9311 from 871 queries. Hiding confidence and returning only labels still recovers the binary model at 0.9900, but over 800 queries instead of 9.

Rounding and noise reduce the fidelity of equation solving in this implementation because the solver does not adapt to perturbation. The adaptive attacker in the paper recovers most of that loss.

## Project Layout

| Path | Contents |
|---|---|
| `backend/victim.py` | Datasets, victim training, defenses |
| `backend/attacker.py` | The four attacks |
| `backend/app.py` | FastAPI app and attack dispatch |
| `backend/testE2E.py` | End-to-end check |
| `frontend/src/App.jsx` | Dashboard |
| `frontend/src/api.js` | API client |

## Reference

F. Tramèr, F. Zhang, A. Juels, M. Reiter, and T. Ristenpart, Stealing Machine Learning Models via Prediction APIs, Proc. 25th USENIX Security Symposium, Austin, 2016, pages 601 to 618. The paper is available at https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer.
