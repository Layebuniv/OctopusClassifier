"""
octopus_classifier.py
=====================
OctopusClassifier — sklearn wrapper of the original formulation.

All internal logic is preserved verbatim from the original
``MST_octopus_classifier_simpleBest`` function:

  - Full pairwise distance matrix (scipy pdist / squareform)
  - MST built with networkx.minimum_spanning_tree on a complete graph
  - Embeddedness = mean edge weight of the closest node's MST neighbours
  - Speed formula (original):
        core_strength       = sqrt(1 + class_mean * log(1+ni) / d_centroid)
        tentacle_efficiency = tree_length / ni^1.5
        tentacle_agility    = (1 / embeddedness) * tentacle_efficiency
        v                   = core_strength + (1 + tentacle_agility)
        t[c]                = d_min / v

The only changes are the sklearn API wrapper (fit / predict),
standard attributes (classes_, n_features_in_), and input validation.

Dependencies
------------
    numpy, scipy, scikit-learn, networkx
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


def _calculate_embeddedness(T: nx.Graph, node_id: int) -> float:
    """
    Embeddedness of a node in the MST (verbatim from original).

    Defined as the mean weight of all MST edges incident to node_id.
    Lower = tighter embedding = stronger tentacle suction.
    Returns np.inf for isolated nodes (no neighbours in T).

    Parameters
    ----------
    T       : networkx.Graph — Minimum Spanning Tree of the class.
    node_id : int — index of the node (row index in the class array).
    """
    neighbours = list(T.neighbors(node_id))
    if not neighbours:
        return np.inf
    edge_weights = [
        T[node_id][nid]["weight"]
        for nid in neighbours
        if T.has_edge(node_id, nid)
    ]
    return float(np.mean(edge_weights)) if edge_weights else np.inf


class OctopusClassifier(BaseEstimator, ClassifierMixin):
    """
    Octopus Classifier — original formulation, sklearn-compatible.

    Models each class as an "octopus":
      - Body      : class centroid
      - Tentacles : Minimum Spanning Tree of all class training points
                    (built with networkx on the full pairwise distance graph)
      - Speed     : core strength × centroid proximity
                    + tentacle agility × MST topology
      - Decision  : the class whose octopus arrives first wins

    Arrival time for class c at test point x::

        d_min               = Euclidean distance to nearest training point
        embeddedness        = mean MST edge weight of that nearest node
        d_centroid          = distance from x to class centroid
        core_strength       = sqrt(1 + class_mean * log(1+ni) / d_centroid)
        tentacle_efficiency = tree_length / ni^1.5
        tentacle_agility    = tentacle_efficiency / embeddedness
        v                   = core_strength + (1 + tentacle_agility)
        t[c]                = d_min / v

    Parameters
    ----------
    None — original formulation has no hyperparameters.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during fit.
    octopus_data_ : list of dict
        Precomputed per-class octopus structures.
    n_features_in_ : int
        Number of features seen during fit.

    Notes
    -----
    Fit complexity is O(n² log n) per class (full pairwise graph + MST).
    For large datasets use FastOctopusClassifier or BestOctopusClassifier.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> clf = make_pipeline(StandardScaler(), OctopusClassifier())
    >>> cross_val_score(clf, *load_iris(return_X_y=True), cv=5).mean()
    0.96
    """

    def fit(self, X, y):
        """
        Fit the Octopus Classifier.

        Builds one octopus per class: centroid, full pairwise networkx MST,
        tree length.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : OctopusClassifier
        """
        X, y = check_X_y(X, y)
        X    = X.astype(float)

        self.classes_       = np.unique(y)
        self.n_features_in_ = X.shape[1]
        N                   = len(y)
        self.class_mean_    = N / len(self.classes_)   # average class size

        self.octopus_data_ = []

        for c in self.classes_:
            Xi = X[y == c]
            ni = Xi.shape[0]

            if ni == 0:
                self.octopus_data_.append({"empty": True})
                continue

            centroid = Xi.mean(axis=0)

            if ni > 1:
                D = squareform(pdist(Xi))

                # Build complete weighted graph — verbatim from original
                G = nx.Graph()
                for ii in range(ni):
                    for jj in range(ii + 1, ni):
                        G.add_edge(ii, jj, weight=D[ii, jj])

                T           = nx.minimum_spanning_tree(G)
                tree_length = sum(nx.get_edge_attributes(T, "weight").values())
            else:
                T           = nx.Graph()
                tree_length = 0.0

            self.octopus_data_.append({
                "empty":       False,
                "Xi":          Xi,
                "centroid":    centroid,
                "ni":          ni,
                "tree_length": tree_length,
                "T":           T,
            })

        return self

    def predict(self, X):
        """
        Predict class labels. Fastest octopus wins.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "octopus_data_")
        X   = check_array(X).astype(float)
        out = np.empty(len(X), dtype=self.classes_.dtype)

        for j, x in enumerate(X):
            t_arrival = np.full(len(self.classes_), np.inf)

            for i, oct_ in enumerate(self.octopus_data_):
                if oct_["empty"]:
                    continue

                Xi          = oct_["Xi"]
                centroid    = oct_["centroid"]
                ni          = oct_["ni"]
                tree_length = oct_["tree_length"]
                T           = oct_["T"]

                # Closest point — verbatim from original
                dists       = np.linalg.norm(Xi - x, axis=1)
                closest_idx = int(np.argmin(dists))
                d_min       = dists[closest_idx]

                # Embeddedness — verbatim from original
                embeddedness = _calculate_embeddedness(T, closest_idx)

                # Core strength — verbatim from original
                d_centroid    = np.linalg.norm(x - centroid)
                core_strength = np.sqrt(
                    1.0 + self.class_mean_ * np.log(1.0 + ni)
                    / (d_centroid + 1e-12)
                )

                # Tentacle agility — verbatim from original
                tentacle_efficiency = tree_length / (ni ** 1.5 + 1e-12)
                tentacle_agility    = (1.0 / (embeddedness + 1e-12)) * tentacle_efficiency

                # Speed and arrival time — verbatim from original
                v              = core_strength + (1.0 + tentacle_agility)
                t_arrival[i]   = d_min / (v + 1e-12)

            out[j] = self.classes_[int(np.argmin(t_arrival))]

        return out
