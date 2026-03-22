"""
fast_octopus_classifier.py
==========================
FastOctopusClassifier — approximate k-NN graph MST with embeddedness.

Replaces the O(n²) full pairwise MST with a sparse k-NN graph MST,
making fit() tractable for large datasets. Adds the embeddedness node
metric to the speed formula.

Speed formula::

    core_strength    = sqrt(1 + class_mean * log(1+ni) / d_centroid)
    tentacle_agility = tree_length / (ni^1.5 * embeddedness)
    v                = core_strength + 1 + tentacle_agility
    t[c]             = d_min / v

Note
----
This version carries the lower-triangular MST bug present in the
kNN-MST session: scipy.minimum_spanning_tree() returns a lower-
triangular matrix, so embeddedness looks at both row and column to
compensate — but this is still unreliable for nodes whose edge falls
in the upper triangle. The bug is fixed in ImprovedOctopusClassifier.

Dependencies
------------
    numpy, scipy, scikit-learn
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


class FastOctopusClassifier(BaseEstimator, ClassifierMixin):
    """
    Fast Octopus Classifier — k-NN graph MST approximation.

    Uses a sparse k-NN graph as input to the MST algorithm, reducing
    fit complexity from O(n²) to O(n·k·log n). Adds the embeddedness
    node metric: mean edge weight of the closest point's MST neighbours.

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbours for k-NN graph construction.
        Increase if you see disconnected-graph warnings.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    octopus_data_ : list of dict
    n_features_in_ : int

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> clf = make_pipeline(StandardScaler(), FastOctopusClassifier())
    >>> cross_val_score(clf, *load_iris(return_X_y=True), cv=5).mean()
    0.96
    """

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_knn_graph(self, X: np.ndarray) -> csr_matrix:
        """Build symmetric sparse k-NN graph."""
        n = len(X)
        if n == 1:
            return csr_matrix((1, 1))
        k = min(self.n_neighbors, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
        dists, idxs = nbrs.kneighbors(X)
        rows, cols, data = [], [], []
        for i in range(n):
            for j in range(1, k + 1):
                rows.append(i)
                cols.append(idxs[i, j])
                data.append(dists[i, j])
        g = csr_matrix((data, (rows, cols)), shape=(n, n))
        return g.minimum(g.T)

    def _embeddedness(self, mst: np.ndarray, node_id: int) -> float:
        """
        Mean edge weight of node_id in the MST.
        Searches both row and column to partially compensate for
        the lower-triangular storage format.
        """
        row_nbrs = np.where(mst[node_id, :] > 0)[0]
        col_nbrs = np.where(mst[:, node_id] > 0)[0]
        nbrs     = np.unique(np.concatenate([row_nbrs, col_nbrs]))
        if len(nbrs) == 0:
            return np.inf
        weights = [max(mst[node_id, n], mst[n, node_id]) for n in nbrs]
        return float(np.mean(weights))

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit using k-NN graph MST approximation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : FastOctopusClassifier
        """
        X, y = check_X_y(X, y)
        X    = X.astype(float)

        self.classes_       = np.unique(y)
        self.n_features_in_ = X.shape[1]
        N                   = len(y)
        self.class_mean_    = N / len(self.classes_)

        self.octopus_data_ = []

        for c in self.classes_:
            Xi = X[y == c]
            ni = len(Xi)

            if ni == 0:
                self.octopus_data_.append({"empty": True})
                continue

            centroid = Xi.mean(axis=0)

            if ni > 1:
                mst         = minimum_spanning_tree(self._build_knn_graph(Xi))
                tree_length = mst.sum()
                mst_dense   = mst.toarray()   # lower-triangular (known limitation)
            else:
                tree_length = 0.0
                mst_dense   = np.zeros((1, 1))

            self.octopus_data_.append({
                "empty":       False,
                "Xi":          Xi,
                "centroid":    centroid,
                "ni":          ni,
                "tree_length": tree_length,
                "mst":         mst_dense,
            })

        return self

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

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
        eps = np.finfo(float).eps
        out = np.empty(len(X), dtype=self.classes_.dtype)

        for j, x in enumerate(X):
            t = np.full(len(self.classes_), np.inf)

            for i, oct_ in enumerate(self.octopus_data_):
                if oct_["empty"]:
                    continue

                Xi          = oct_["Xi"]
                ni          = oct_["ni"]
                tree_length = oct_["tree_length"]
                centroid    = oct_["centroid"]
                mst         = oct_["mst"]

                dists       = np.linalg.norm(Xi - x, axis=1)
                d_min       = dists.min()
                closest_idx = int(dists.argmin())

                emb = self._embeddedness(mst, closest_idx)
                d_c = np.linalg.norm(x - centroid)

                core_strength    = np.sqrt(
                    1.0 + self.class_mean_ * np.log1p(ni) / (d_c + eps)
                )
                tentacle_agility = tree_length / (ni ** 1.5 + eps) / (emb + eps)
                v                = core_strength + 1.0 + tentacle_agility
                t[i]             = d_min / (v + eps)

            out[j] = self.classes_[t.argmin()]

        return out
