# MST Octopus Classifier

A family of novel classifiers that model each class as an **octopus** — a creature with a body (centroid) and tentacles (Minimum Spanning Tree) that races toward every test point. The class whose octopus arrives first wins.
# Citation
 
If you use this work in your research, please cite the original paper:
 
> Layeb, A. & Khadoudja, G. (2025). **Octopus Classifier: A New Bio-inspired Classifier Based Minimal Spanning Tree**. In *2025 Fourth International Conference on Theoretical and Applicative Aspects of Computer Science (ICTAACS)* (pp. 1–7). IEEE.  
> DOI: [10.1109/ICTAACS64990.2025.11399349](https://ieeexplore.ieee.org/document/11399349)
 
BibTeX:
 
```bibtex
@inproceedings{layeb2025octopus,
  author    = {Layeb, Abdesslem and Khadoudja, Ghanem},
  title     = {Octopus Classifier: A New Bio-inspired Classifier Based Minimal Spanning Tree},
  booktitle = {2025 Fourth International Conference on Theoretical and Applicative Aspects of Computer Science (ICTAACS)},
  pages     = {1--7},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICTAACS64990.2025.11399349},
  url       = {https://ieeexplore.ieee.org/document/11399349}
}

## The idea

Each class is represented by an octopus:

| Component | Meaning |
|-----------|---------|
| **Body** | Class centroid |
| **Tentacles** | Minimum Spanning Tree over all training points in the class |
| **Speed** | `core_strength + 1 + tentacle_agility` |
| **Decision** | `argmin_c  t[c]`  where  `t[c] = d_min / speed` |

```
core_strength       = sqrt(1 + class_mean × log(1+n) / d_centroid)
tentacle_efficiency = tree_length / n^1.5
tentacle_agility    = tentacle_efficiency / embeddedness
embeddedness        = mean MST edge weight of the nearest training node
```

## Classifiers

| Class | File | MST method | Notes |
|-------|------|-----------|-------|
| `OctopusClassifier` | `octopus_classifier.py` | Full pairwise — networkx | Original formulation |
| `FastOctopusClassifier` | `fast_octopus_classifier.py` | k-NN graph — scipy | Approximate; scales to large n |
| `ImprovedOctopusClassifier` | `improved_octopus_classifier.py` | Exact ≤ 300, k-NN otherwise | 8 structural fixes (V2) |
| `BestOctopusClassifier` | `best_octopus_classifier.py` | Exact ≤ 300, k-NN otherwise | Adaptive prior + validated defaults (V4) |

All classifiers are **sklearn-compatible**: `fit` / `predict`, work inside `Pipeline`, `GridSearchCV`, `cross_val_score`.

## Benchmark results

20 datasets · 5-fold stratified CV · StandardScaler fitted inside each fold

| Classifier | Mean Acc | Wins / 20 | ≥ 95% |
|---|---|---|---|
| **BestOctopusClassifier** | **95.54 %** | **15** | **14** |
| ImprovedOctopusClassifier | 94.52 % | 6 | 10 |
| OctopusClassifier | 90.41 % | 1 | 6 |
| FastOctopusClassifier | 90.31 % | 1 | 6 |
| SVM (rbf) | 92.64 % | 3 | 9 |
| RandomForest (100) | 89.60 % | 1 | 4 |
| NaiveBayes | 83.50 % | 0 | 2 |

## Installation

```bash
pip install numpy scipy scikit-learn networkx
git clone https://github.com/YOUR_USERNAME/mst-octopus-classifier
cd mst-octopus-classifier
```

## Quick start

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import numpy as np

from best_octopus_classifier import BestOctopusClassifier

X, y = load_wine(return_X_y=True)

skf  = StratifiedKFold(5, shuffle=True, random_state=0)
accs = []
for tr, te in skf.split(X, y):
    sc  = StandardScaler()
    clf = BestOctopusClassifier().fit(sc.fit_transform(X[tr]), y[tr])
    accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))

print(f"Wine 5-fold CV: {np.mean(accs):.4f}")   # → 0.9717
```

Calibrated probabilities and confidence-based rejection:

```python
proba = clf.predict_proba(X_test, tau=1.0)          # (n_samples, n_classes)

labels, conf = clf.predict_with_confidence(X_test)
reliable     = labels[conf > 0.70]                  # abstain when uncertain
```

## Choosing a classifier

| Use case | Recommended |
|----------|-------------|
| Best accuracy | `BestOctopusClassifier` |
| Studying the original idea | `OctopusClassifier` |
| Large dataset (n > 500/class) | `FastOctopusClassifier` |
| Understanding the fixes | `ImprovedOctopusClassifier` |

## Run the benchmark

```bash
python benchmark.py
```

Runs all four Octopus variants plus SVM, RandomForest, and NaiveBayes on 20 datasets. Results printed to stdout and saved to `results.json`.

## Development lineage

```
OctopusClassifier          Original. Full pairwise MST via networkx.
        │
        ▼
FastOctopusClassifier      k-NN graph MST (scipy). Adds embeddedness.
        │                  Lower-triangular MST bug present.
        ▼
ImprovedOctopusClassifier  8 structural fixes. +4.11 pp over Original.
        ▼
BestOctopusClassifier      Adaptive prior weight per class.
                           Grid-validated defaults (pw=0.2, k=10).
                           +1.02 pp over Improved. Beats SVM overall.
```

## The 8 fixes in ImprovedOctopusClassifier

| # | Bug | Fix |
|---|-----|-----|
| 1 | MST lower-triangular only — half edges invisible | `raw + raw.T` |
| 2 | `core_strength → ∞` when test point ≈ centroid | Clamp with per-class `sigma` |
| 3 | `tree_length / ni^1.5` unbounded on chain classes | Use mean edge weight |
| 4 | k-NN MST misses long-range edges | Full pairwise MST for `ni ≤ 300` |
| 5 | Single nearest point is noisy | Average `k=3` nearest points |
| 6 | Euclidean distance ignores correlations | Mahalanobis per-class distance |
| 7 | Majority classes dominate imbalanced data | `(1/prior)^pw` handicap |
| 8 | O(n·d) brute-force NN at predict time | O(log n) BallTree index |

## The V4 innovation in BestOctopusClassifier

Fixed prior weight treats all classes identically. Adaptive prior scales by class size:

```
pw_c = prior_weight / log(1 + ni_c)
t[c] *= (1 / prior_c) ^ pw_c
```

Large class → small exponent → gentle handicap.  
Small class → large exponent → strong protection.  
A massive octopus is slowed proportionally to its own bulk — making the metaphor self-consistent.

## License

MIT — see `LICENSE`.
