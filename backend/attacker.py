"""Attacker library: paper-faithful but compact reimplementation.

Attack 1 equation solving for binary LR: d+1 queries, solve w.x+b=logit(p).
Attack 2 retraining with soft labels for softmax LR and MLP.
Attack 3 path-finding lite for decision trees using (label, proba) leaf ids.
Attack 4 label-only adaptive baseline inspired by Lowd-Meek.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

LOGIT_CLIP = 1e-6


def logit(p):
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def random_points(n, d, rng):
    return rng.uniform(-1, 1, size=(n, d))


def fidelity(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# --- Attack 1: exact equation solving, binary LR ---
def attack_binary_lr(predict_proba, d, rng):
    n = d + 1
    X = random_points(n, d, rng)
    p = np.array([predict_proba(x)[1] for x in X])
    A = np.hstack([X, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(A, logit(p), rcond=None)
    return {"w": sol[:-1], "b": float(sol[-1]), "queries": n}


def eval_linear(w, b, X):
    return (X @ w + b >= 0).astype(int)


# --- Attack 2: retraining on soft labels ---
def attack_retrain(predict_fn, d, n_classes, budget, rng, kind="softmax"):
    X = random_points(budget, d, rng)
    P = np.array([predict_fn(x) for x in X])
    y_soft = P.argmax(axis=1)
    if kind == "mlp":
        classifier = MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, random_state=1)
    else:
        classifier = LogisticRegression(multi_class="multinomial", max_iter=2000)
    classifier.fit(X, y_soft)
    
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        subsetSize = max(10, int(budget * frac))
        subsetClassifier = LogisticRegression(multi_class="multinomial", max_iter=2000) if kind != "mlp" else \
            MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, random_state=1)
        subsetClassifier.fit(X[:subsetSize], y_soft[:subsetSize])
        curve.append({"queries": subsetSize, "train_fit": float(subsetClassifier.score(X[:subsetSize], y_soft[:subsetSize]))})
    return {"model": classifier, "Xq": X, "curve": curve, "queries": budget}


# --- Attack 3: tree path-finding lite ---
def leaf_identifier(label, proba):
    maxProba = float(np.max(proba)) if proba is not None else -1.0
    return f"{int(label)}@{round(maxProba, 6)}"


def attack_tree(predict_fn, numFeatures, budget, rng, max_depth=5):
    X_test_pool = random_points(min(budget, 4000), numFeatures, rng)
    seenLeaves = {}
    leafLabels = {}
    for x in X_test_pool:
        label, proba = predict_fn(x)
        leafId = leaf_identifier(label, proba)
        seenLeaves.setdefault(leafId, []).append(x)
        leafLabels[leafId] = int(label)
        if len(seenLeaves) >= 2 ** (max_depth + 1):
            break
    queries_used = len(X_test_pool)
    
    thresholds = {}
    probes_per_feature = 15
    basePoint = rng.uniform(-1, 1, size=(numFeatures,))
    base_label, base_proba = predict_fn(basePoint)
    base_id = leaf_identifier(base_label, base_proba)
    queries_used += 1
    for featureIndex in range(numFeatures):
        foundThresholds = []
        for featureValue in np.linspace(-1, 1, probes_per_feature):
            x = basePoint.copy()
            x[featureIndex] = featureValue
            label, proba = predict_fn(x)
            queries_used += 1
            if leaf_identifier(label, proba) != base_id:
                foundThresholds.append(round(float(featureValue), 3))
                if len(foundThresholds) >= 3:
                    break
        if foundThresholds:
            thresholds[f"x{featureIndex}"] = foundThresholds
            
    X = np.array([x for pts in seenLeaves.values() for x in pts])
    y = np.array([leafLabels[lid] for lid, pts in seenLeaves.items() for _ in pts])
    
    surrogate = DecisionTreeClassifier(max_depth=None, random_state=1).fit(X, y)
    return {"model": surrogate, "n_leaves": len(seenLeaves),
            "thresholds": thresholds, "queries": int(queries_used)}


# --- Attack 4: label-only baseline (uniform retraining + boundary search) ---
def attack_label_only(predict_label, d, budget, rng, kind="softmax"):
    X = random_points(budget, d, rng)
    y = np.array([predict_label(x) for x in X])
    if kind == "mlp":
        classifier = MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, random_state=1).fit(X, y)
    elif kind == "tree":
        classifier = DecisionTreeClassifier(max_depth=5, random_state=1).fit(X, y)
    else:
        classifier = LogisticRegression(multi_class="auto", max_iter=2000).fit(X, y)
    return {"model": classifier, "queries": budget}
