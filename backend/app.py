"""Victim prediction API + attack runner. Run: uvicorn app:app --reload"""
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.tree import _tree

from victim import train_victims, apply_defense, DEFENSES
from attacker import (attack_binary_lr, attack_retrain, attack_tree,
                      attack_label_only, eval_linear, fidelity)

app = FastAPI(title="Model Extraction Lab API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STORE = train_victims()
QUERY_COUNT = {"n": 0}


def _predict(model_id, x, defense="full"):
    modelEntry = STORE[model_id]
    model = modelEntry["model"]
    probabilities = model.predict_proba([x])[0]
    defendedProbabilities = apply_defense([probabilities], defense)
    predictedLabel = int(np.argmax(probabilities))
    QUERY_COUNT["n"] += 1
    if defendedProbabilities is None:
        return predictedLabel, None
    return predictedLabel, [float(probability) for probability in defendedProbabilities[0]]



class PredictIn(BaseModel):
    model_id: str
    x: list
    defense: str = "full"


class AttackIn(BaseModel):
    model_id: str
    defense: str = "full"
    budget: int = 800


@app.get("/models")
def models():
    availableModels = {}
    for modelId, modelEntry in STORE.items():
        model = modelEntry["model"]
        accuracy = float(model.score(modelEntry["X_test"], modelEntry["y_test"]))
        availableModels[modelId] = {"desc": modelEntry["desc"], "kind": modelEntry["kind"], "features": modelEntry["features"],
                  "n_features": len(modelEntry["features"]), "test_acc": round(accuracy, 4),
                  "defenses": list(DEFENSES)}
    return {"models": availableModels, "queries_served": QUERY_COUNT["n"]}


@app.post("/predict")
def predict(inp: PredictIn):
    label, proba = _predict(inp.model_id, inp.x, inp.defense)
    return {"label": label, "proba": proba}


def _tree_json(model, features):
    treeModel = model.tree_
    def node(index):
        if treeModel.feature[index] == _tree.TREE_UNDEFINED:
            nodeValues = treeModel.value[index][0]
            return {"leaf": True, "label": int(nodeValues.argmax()),
                    "samples": int(treeModel.n_node_samples[index]),
                    "proba": [round(float(probability), 3) for probability in nodeValues / nodeValues.sum()]}
        return {"leaf": False, "feature": features[treeModel.feature[index]],
                "threshold": round(float(treeModel.threshold[index]), 3),
                "left": node(treeModel.children_left[index]), "right": node(treeModel.children_right[index])}
    return node(0)


@app.get("/tree/{model_id}")
def tree(model_id: str):
    modelEntry = STORE[model_id]
    if modelEntry["kind"] != "tree":
        return {"error": "not a tree model"}
    return {"tree": _tree_json(modelEntry["model"], modelEntry["features"])}


@app.post("/attack")
def attack(inp: AttackIn):
    rng = np.random.default_rng(7)
    modelEntry = STORE[inp.model_id]
    modelType, numFeatures = modelEntry["kind"], len(modelEntry["features"])
    defense = inp.defense

    def proba_fn(x):
        _, probabilities = _predict(inp.model_id, list(x), defense)
        if probabilities is None:  # label-only fallback: one-hot
            label, _ = _predict(inp.model_id, list(x), "full")
            numClasses = len(modelEntry["model"].classes_)
            oneHot = [0.0] * numClasses
            oneHot[label] = 1.0
            return oneHot
        return probabilities

    def full_proba_fn(x):
        _, probabilities = _predict(inp.model_id, list(x), "full")
        return probabilities

    def label_fn(x):
        label, _ = _predict(inp.model_id, list(x), defense)
        return label

    queriesBefore = QUERY_COUNT["n"]
    attackSummary = {}
    if modelType == "binary_lr" and defense in ("full", "rounded", "noisy"):
        attackOutput = attack_binary_lr(lambda x: proba_fn(x), numFeatures, rng)
        predictions = eval_linear(attackOutput["w"], attackOutput["b"], modelEntry["X_test"])
        trueLabels = modelEntry["model"].predict(modelEntry["X_test"])
        trueWeights = modelEntry["model"].coef_[0]
        attackSummary = {"attack": "Equation Solving", "queries": attackOutput["queries"],
                  "fidelity": round(fidelity(trueLabels, predictions), 4),
                  "w_error": round(float(np.linalg.norm(trueWeights - attackOutput["w"])), 6),
                  "b_error": round(float(abs(modelEntry["model"].intercept_[0] - attackOutput["b"])), 6),
                  "extracted": {"w": [round(float(weight), 4) for weight in attackOutput["w"]],
                                "b": round(attackOutput["b"], 4)},
                  "victim": {"w": [round(float(weight), 4) for weight in trueWeights],
                             "b": round(float(modelEntry["model"].intercept_[0]), 4)},
                  "note": "Nine queries are enough to solve the linear equation exactly."}
    elif modelType == "tree" and defense != "label_only":
        attackOutput = attack_tree(lambda x: (_predict(inp.model_id, list(x), defense)),
                        numFeatures, min(inp.budget, 4000), rng)
        predictions = attackOutput["model"].predict(modelEntry["X_test"])
        trueLabels = modelEntry["model"].predict(modelEntry["X_test"])
        attackSummary = {"attack": "Path Finding", "queries": attackOutput["queries"],
                  "fidelity": round(fidelity(trueLabels, predictions), 4),
                  "n_leaves_found": attackOutput["n_leaves"],
                  "thresholds": attackOutput["thresholds"],
                  "note": "The label and confidence act as a unique identifier for the leaf. A line search finds the split thresholds."}
    else:
        # softmax / mlp with confidences -> soft retraining; else label-only baseline
        surrogateType = "mlp" if modelType == "mlp" else ("tree" if modelType == "tree" else "softmax")
        if defense == "label_only":
            attackOutput = attack_label_only(label_fn, numFeatures, inp.budget, rng, kind=surrogateType)
            attackName = "Label Only Retraining"
        else:
            attackOutput = attack_retrain(lambda x: full_proba_fn(x) if defense == "full"
                               else proba_fn(x), numFeatures,
                               len(modelEntry["model"].classes_), inp.budget, rng,
                               kind="mlp" if modelType == "mlp" else "softmax")
            attackName = "Soft Label Retraining"
        predictions = attackOutput["model"].predict(modelEntry["X_test"])
        trueLabels = modelEntry["model"].predict(modelEntry["X_test"])
        attackSummary = {"attack": attackName, "queries": attackOutput["queries"],
                  "fidelity": round(fidelity(trueLabels, predictions), 4),
                  "curve": attackOutput.get("curve", []),
                  "note": "The attacker trains a second model using the queried labels."}
    attackSummary["queries_used_this_attack"] = QUERY_COUNT["n"] - queriesBefore
    attackSummary["model_id"] = inp.model_id
    attackSummary["defense"] = defense
    return attackSummary
