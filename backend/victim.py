"""Victim models on custom synthetic data.

Two story datasets:
- fraud: 8-feature tabular transaction data, binary label (legit/fraud)
- triage: 6-feature multiclass data with 3 classes
Features are scaled to [-1, 1] so equation-solving math is clean.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

RNG = 42

FRAUD_FEATURES = ["amount", "hour", "merchant_score", "device_trust",
                  "velocity_1h", "geo_distance", "age_days", "fail_rate"]
TRIAGE_FEATURES = ["f0", "f1", "f2", "f3", "f4", "f5"]

DEFENSES = ("full", "rounded", "label_only", "noisy")


def _scale(features):
    minVals, maxVals = features.min(axis=0), features.max(axis=0)
    return 2 * (features - minVals) / np.maximum(maxVals - minVals, 1e-9) - 1


def build_datasets():
    fraudFeatures, fraudLabels = make_classification(n_samples=6000, n_features=8, n_informative=6,
                                 n_redundant=1, flip_y=0.01, class_sep=1.2,
                                 random_state=RNG)
    triageFeatures, triageLabels = make_classification(n_samples=6000, n_features=6, n_informative=5,
                                 n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                 class_sep=1.4, random_state=RNG + 1)
    return (_scale(fraudFeatures), fraudLabels), (_scale(triageFeatures), triageLabels)


def train_victims():
    (fraudFeatures, fraudLabels), (triageFeatures, triageLabels) = build_datasets()
    fraudFeaturesTrain, fraudFeaturesTest, fraudLabelsTrain, fraudLabelsTest = train_test_split(fraudFeatures, fraudLabels, test_size=0.3, random_state=RNG)
    triageFeaturesTrain, triageFeaturesTest, triageLabelsTrain, triageLabelsTest = train_test_split(triageFeatures, triageLabels, test_size=0.3, random_state=RNG)

    fraud_lr = LogisticRegression(max_iter=2000).fit(fraudFeaturesTrain, fraudLabelsTrain)
    multi_lr = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                  max_iter=2000).fit(triageFeaturesTrain, triageLabelsTrain)
    mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=800,
                        random_state=RNG).fit(triageFeaturesTrain, triageLabelsTrain)
    tree = DecisionTreeClassifier(max_depth=5, random_state=RNG).fit(fraudFeaturesTrain, fraudLabelsTrain)

    return {
        "fraud_lr": {"model": fraud_lr, "X_test": fraudFeaturesTest, "y_test": fraudLabelsTest,
                     "features": FRAUD_FEATURES, "kind": "binary_lr",
                     "desc": "Fraud Lr On 8 Features Of Synthetic Fraud Data"},
        "multi_lr": {"model": multi_lr, "X_test": triageFeaturesTest, "y_test": triageLabelsTest,
                     "features": TRIAGE_FEATURES, "kind": "softmax",
                     "desc": "Multi-Class Softmax On 6 Features Of Synthetic Triage Data"},
        "mlp": {"model": mlp, "X_test": triageFeaturesTest, "y_test": triageLabelsTest,
                "features": TRIAGE_FEATURES, "kind": "mlp",
                "desc": "Mlp With One Hidden Layer And 2043 Parameters"},
        "tree": {"model": tree, "X_test": fraudFeaturesTest, "y_test": fraudLabelsTest,
                 "features": FRAUD_FEATURES, "kind": "tree",
                 "desc": "Decision Tree Depth 5 On Fraud Data"},
    }


def apply_defense(probabilities, defense, rng=None):
    probabilities = np.asarray(probabilities, dtype=float)
    if defense == "label_only":
        return None
    if defense == "rounded":
        return np.round(probabilities, 2)
    if defense == "noisy":
        rng = rng or np.random.default_rng(0)
        noisy = probabilities + rng.normal(0, 0.05, size=probabilities.shape)
        noisy = np.clip(noisy, 1e-6, 1 - 1e-6)
        return noisy / noisy.sum(axis=-1, keepdims=True)
    return probabilities
