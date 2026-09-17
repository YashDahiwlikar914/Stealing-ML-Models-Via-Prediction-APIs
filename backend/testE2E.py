import numpy as np
from victim import train_victims
from attacker import attack_binary_lr, attack_retrain, attack_tree, attack_label_only, eval_linear, fidelity

store = train_victims()
rng = np.random.default_rng(0)

# 1. Binary LR exact
fraudEntry = store["fraud_lr"]
fraudModel = fraudEntry["model"]
fraudAttack = attack_binary_lr(lambda x: fraudModel.predict_proba([x])[0], 8, rng)
fraudPredictions = eval_linear(fraudAttack["w"], fraudAttack["b"], fraudEntry["X_test"])
print("binary_lr fidelity:", fidelity(fraudModel.predict(fraudEntry["X_test"]), fraudPredictions), "queries:", fraudAttack["queries"])

# 2. Softmax retrain
multiEntry = store["multi_lr"]
multiModel = multiEntry["model"]
multiAttack = attack_retrain(lambda x: multiModel.predict_proba([x])[0], 6, 3, 600, rng, kind="softmax")
print("softmax fidelity:", fidelity(multiModel.predict(multiEntry["X_test"]), multiAttack["model"].predict(multiEntry["X_test"])))

# 3. Tree path finding
treeEntry = store["tree"]
treeModel = treeEntry["model"]
def tree_predict_fn(x):
    probabilities = treeModel.predict_proba([x])[0]
    return int(probabilities.argmax()), probabilities
treeAttack = attack_tree(tree_predict_fn, 8, 4000, rng)
print("tree fidelity:", fidelity(treeModel.predict(treeEntry["X_test"]), treeAttack["model"].predict(treeEntry["X_test"])), "leaves:", treeAttack["n_leaves"])
