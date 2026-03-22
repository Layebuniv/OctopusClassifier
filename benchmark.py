"""
benchmark.py
============
20-dataset benchmark: all four MST Octopus classifiers vs SVM, RF, NaiveBayes.

Datasets
--------
4 sklearn builtins:   Iris, Wine, BreastCancer, Digits
16 UCI-characteristic synthetic proxies (seeded make_classification):
    Glass, Ionosphere, Sonar, Parkinsons, HeartStatlog, Diabetes,
    Vehicle, BalanceScale, Seeds, Haberman, Ecoli, LiverDisorders,
    BloodTransfusion, Banknote, Wholesale, Yeast

Protocol
--------
5-fold stratified CV. StandardScaler fitted inside each fold (no leakage).
Results saved to results.json.

Usage
-----
    python benchmark.py
"""

import numpy as np
import warnings
import json
import time

warnings.filterwarnings("ignore")

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, make_classification
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from octopus_classifier import OctopusClassifier
from fast_octopus_classifier import FastOctopusClassifier
from improved_octopus_classifier import ImprovedOctopusClassifier
from best_octopus_classifier import BestOctopusClassifier


# =============================================================================
# Datasets
# =============================================================================

def get_datasets() -> dict:
    ds = {}
    ds["Iris"]         = load_iris(return_X_y=True)
    ds["Wine"]         = load_wine(return_X_y=True)
    ds["BreastCancer"] = load_breast_cancer(return_X_y=True)
    ds["Digits"]       = load_digits(return_X_y=True)

    specs = [
        ("Glass",            214,  9,  7,  6, 10),
        ("Ionosphere",       351, 33, 20,  2, 11),
        ("Sonar",            208, 60, 40,  2, 12),
        ("Parkinsons",       195, 22, 15,  2, 13),
        ("HeartStatlog",     270, 13, 10,  2, 14),
        ("Diabetes",         768,  8,  5,  2, 15),
        ("Vehicle",          846, 18, 14,  4, 16),
        ("BalanceScale",     625,  4,  4,  3, 17),
        ("Seeds",            210,  7,  6,  3, 18),
        ("Haberman",         306,  3,  3,  2, 19),
        ("Ecoli",            336,  7,  5,  8, 20),
        ("LiverDisorders",   345,  6,  5,  2, 21),
        ("BloodTransfusion", 748,  4,  4,  2, 22),
        ("Banknote",        1372,  4,  4,  2, 23),
        ("Wholesale",        440,  7,  6,  3, 24),
        ("Yeast",           1484,  8,  6, 10, 25),
    ]
    for name, n, feats, info, classes, seed in specs:
        X, y = make_classification(
            n_samples=n, n_features=feats, n_informative=info,
            n_redundant=max(0, feats - info - 1), n_repeated=0,
            n_classes=classes, n_clusters_per_class=1, random_state=seed,
        )
        ds[name] = (X.astype(float), y)
    return ds


# =============================================================================
# CV
# =============================================================================

def cv_eval(clf_fn, X, y, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        sc       = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf      = clf_fn()
        clf.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs)), float(np.std(accs))


# =============================================================================
# Main
# =============================================================================

def main():
    datasets = get_datasets()
    print(f"Loaded {len(datasets)} datasets\n", flush=True)

    classifiers = {
        "OctopusClassifier":         lambda: OctopusClassifier(),
        "FastOctopusClassifier":     lambda: FastOctopusClassifier(),
        "ImprovedOctopusClassifier": lambda: ImprovedOctopusClassifier(),
        "BestOctopusClassifier":     lambda: BestOctopusClassifier(),
        "SVM":                       lambda: SVC(kernel="rbf"),
        "RandomForest":              lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "NaiveBayes":                lambda: GaussianNB(),
    }

    short = {
        "OctopusClassifier":         "Octopus",
        "FastOctopusClassifier":     "Fast",
        "ImprovedOctopusClassifier": "Improved",
        "BestOctopusClassifier":     "Best",
        "SVM":                       "SVM",
        "RandomForest":              "RF",
        "NaiveBayes":                "NB",
    }

    results = {c: [] for c in classifiers}
    std_map = {c: [] for c in classifiers}
    times   = {c: [] for c in classifiers}
    rows    = []

    col_w  = 11
    header = f"{'Dataset':<22}" + "".join(f"{short[c]:>{col_w}}" for c in classifiers)
    print(header)
    print("=" * (22 + col_w * len(classifiers)))

    for name, (X, y) in datasets.items():
        X   = X.astype(np.float64)
        row = {"dataset": name}

        for cname, fn in classifiers.items():
            is_oct = cname in (
                "OctopusClassifier", "FastOctopusClassifier",
                "ImprovedOctopusClassifier", "BestOctopusClassifier"
            )
            splits = 3 if (name == "Digits" and is_oct) else 5
            t0 = time.time()
            acc, std = cv_eval(fn, X, y, n_splits=splits)
            results[cname].append(acc)
            std_map[cname].append(std)
            times[cname].append(time.time() - t0)
            row[cname] = acc

        rows.append(row)
        print(f"{name:<22}" + "".join(f"{row[c]:>{col_w}.4f}" for c in classifiers),
              flush=True)

    print("=" * (22 + col_w * len(classifiers)))

    wins     = {c: 0 for c in classifiers}
    above_95 = {c: 0 for c in classifiers}
    for row in rows:
        best = max(row[c] for c in classifiers)
        for c in classifiers:
            if abs(row[c] - best) < 1e-9:
                wins[c] += 1
            if row[c] >= 0.95:
                above_95[c] += 1

    print(f"\n{'Classifier':<26} {'Mean':>8}  {'±Std':>7}  {'Wins':>6}  {'≥95%':>6}  {'Time(s)':>8}")
    print("-" * 68)
    for c in classifiers:
        print(f"{short[c]:<26} {np.mean(results[c]):>8.4f}  "
              f"{np.mean(std_map[c]):>7.4f}  "
              f"{wins[c]:>6}  "
              f"{above_95[c]:>6}  "
              f"{np.sum(times[c]):>8.1f}")

    output = {
        "datasets":    [r["dataset"] for r in rows],
        "classifiers": list(classifiers.keys()),
        "short_names": short,
        "accs":        {c: results[c] for c in classifiers},
        "means":       {c: float(np.mean(results[c])) for c in classifiers},
        "stds":        {c: float(np.mean(std_map[c]))  for c in classifiers},
        "wins":        wins,
        "above_95":    above_95,
        "times":       {c: float(np.sum(times[c])) for c in classifiers},
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results.json")


if __name__ == "__main__":
    main()
