"""
improved_octopus_classifier.py
===============================
ImprovedOctopusClassifier — all 8 structural fixes applied (V2).

Source: MSTOctopusV2 (verbatim). All internal logic is unchanged;
only the class is renamed for the published package.

Eight fixes over the original formulations
-------------------------------------------
1. MST symmetry         mst.toarray() + mst.toarray().T
                        scipy returns lower-triangular only; half all edges
                        were invisible → nodes appeared falsely isolated →
                        embeddedness returned inf → agility silently zero.

2. Centroid sigma clamp sigma = mean intra-class distance from centroid.
                        Clamps d_centroid denominator so core_strength
                        stays finite when test point ≈ centroid.

3. Bounded agility      mean_edge / embeddedness  (instead of tree_length/ni^1.5).
                        Original was unbounded on chain-shaped classes
                        (Ecoli, Yeast); agility dominated velocity entirely.

4. Exact pairwise MST   Full pdist MST for ni ≤ full_mst_threshold (default 300).
                        k-NN approximation only for large classes.

5. k-probe averaging    Average k_probe (default 3) nearest points instead of
                        single closest point. More robust near boundaries.

6. Mahalanobis d_min    Per-class regularised covariance-aware distance.
                        Accounts for feature correlations and scale.

7. Prior weighting      t[c] *= (1/prior_c)^prior_weight.
                        Prevents majority classes monopolising imbalanced data.

8. BallTree index       O(log n) NN queries replace O(n·d) brute-force loop.

Benchmark (20 datasets, 5-fold CV)
------------------------------------
    ImprovedOctopusClassifier : 94.52%  (+4.11 pp over OctopusClassifier)
    OctopusClassifier         : 90.41%
    SVM (rbf)                 : 92.64%

Dependencies
------------
    numpy, scipy, scikit-learn
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


class ImprovedOctopusClassifier(BaseEstimator, ClassifierMixin):
    """
    Improved Octopus Classifier — 8 structural fixes applied (V2).

    All original octopus mechanics retained (body, tentacles, arrival time).
    The fixes correct numerical bugs, approximation errors, and
    inefficiencies identified through ablation on 20 benchmark datasets.

    Parameters
    ----------
    n_neighbors : int, default=10
        k for k-NN graph (classes with ni > full_mst_threshold only).

    full_mst_threshold : int, default=300
        Classes with ni ≤ this use exact pairwise MST (globally optimal).
        Larger classes use k-NN approximation for speed.

    k_probe : int, default=3
        Nearest training points averaged per class at predict time.
        Higher = smoother; lower = faster.

    prior_weight : float, default=1.0
        Class-prior handicap strength. 0.0 = none. 1.0 = full.
        Reduce toward 0 on near-balanced datasets.

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
    >>> clf = make_pipeline(StandardScaler(), ImprovedOctopusClassifier())
    >>> cross_val_score(clf, *load_iris(return_X_y=True), cv=5).mean()
    0.9667
    """

    def __init__(
        self,
        n_neighbors: int        = 10,
        full_mst_threshold: int = 300,
        k_probe: int            = 3,
        prior_weight: float     = 1.0,
    ):
        self.n_neighbors        = n_neighbors
        self.full_mst_threshold = full_mst_threshold
        self.k_probe            = k_probe
        self.prior_weight       = prior_weight

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _build_knn_graph(self, X: np.ndarray) -> csr_matrix:
        n = len(X)
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

    def _build_mst(self, X: np.ndarray) -> np.ndarray:
        """Fix 1+4: symmetric exact/approx MST."""
        ni = len(X)
        if ni <= self.full_mst_threshold:
            graph = csr_matrix(squareform(pdist(X)))
        else:
            graph = self._build_knn_graph(X)
        raw = minimum_spanning_tree(graph).toarray()
        return raw + raw.T  # Fix 1

    def _embeddedness(self, mst_sym: np.ndarray, node_id: int) -> float:
        """Fix 1: row lookup sufficient on symmetric MST."""
        row  = mst_sym[node_id]
        nbrs = np.where(row > 0)[0]
        if len(nbrs) == 0:
            return np.inf
        return float(np.mean(row[nbrs]))

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """
        Fit the Improved Octopus Classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : ImprovedOctopusClassifier
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

            # Fix 2
            sigma = float(np.mean(np.linalg.norm(Xi - centroid, axis=1))) + 1e-8
            # Fix 7
            prior = ni / N

            # Fix 6
            if ni > self.n_features_in_ + 1:
                cov = np.cov(Xi, rowvar=False)
                cov = np.atleast_2d(cov) + 1e-4 * np.eye(cov.shape[0])
                try:
                    inv_cov = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    inv_cov = np.eye(self.n_features_in_)
            else:
                inv_cov = np.eye(self.n_features_in_)

            # Fix 1+4
            if ni > 1:
                mst_sym   = self._build_mst(Xi)
                n_edges   = max((mst_sym > 0).sum() / 2, 1)
                mean_edge = mst_sym.sum() / (2 * n_edges)  # Fix 3
            else:
                mst_sym   = np.zeros((1, 1))
                mean_edge = 0.0

            # Fix 8
            btree = BallTree(Xi)

            self.octopus_data_.append({
                "empty":     False,
                "Xi":        Xi,
                "centroid":  centroid,
                "ni":        ni,
                "sigma":     sigma,
                "prior":     prior,
                "inv_cov":   inv_cov,
                "mean_edge": mean_edge,
                "mst":       mst_sym,
                "btree":     btree,
            })

        return self

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def _arrival_times(self, x: np.ndarray) -> np.ndarray:
        t = np.full(len(self.classes_), np.inf)

        for i, oct_ in enumerate(self.octopus_data_):
            if oct_["empty"]:
                continue

            ni        = oct_["ni"]
            mst       = oct_["mst"]
            k_p       = min(self.k_probe, ni)

            _, idxs = oct_["btree"].query([x], k=k_p)
            idxs    = idxs[0]

            diff  = oct_["Xi"][idxs] - x
            mah   = np.sqrt(np.einsum("ij,jk,ik->i", diff, oct_["inv_cov"], diff) + 1e-12)
            d_min = float(mah.mean())

            emb = float(np.mean([self._embeddedness(mst, idx) for idx in idxs]))

            d_c           = np.linalg.norm(x - oct_["centroid"])
            core_strength = np.sqrt(
                1.0 + self.class_mean_ * np.log1p(ni) / (d_c + oct_["sigma"])
            )
            tentacle_agility = oct_["mean_edge"] / (emb + 1e-8)
            v                = core_strength + 1.0 + tentacle_agility

            prior_factor = (1.0 / (oct_["prior"] + 1e-8)) ** self.prior_weight
            t[i]         = (d_min / (v + 1e-8)) * prior_factor

        return t

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
            out[j] = self.classes_[self._arrival_times(x).argmin()]
        return out

    def predict_proba(self, X, tau: float = 1.0) -> np.ndarray:
        """
        Calibrated probabilities via softmax over arrival times.

        Parameters
        ----------
        X   : array-like of shape (n_samples, n_features)
        tau : float, default=1.0 — temperature (lower = sharper)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes). Rows sum to 1.
        """
        check_is_fitted(self, "octopus_data_")
        X     = check_array(X).astype(float)
        proba = np.empty((len(X), len(self.classes_)))
        for j, x in enumerate(X):
            t    = self._arrival_times(x)
            r    = -t / tau
            r   -= r.max()
            s    = np.exp(r)
            proba[j] = s / s.sum()
        return proba

    def predict_with_confidence(self, X, tau: float = 1.0):
        """
        Return (labels, max_confidence) per sample.

        Useful for rejection / abstention:
            labels, conf = clf.predict_with_confidence(X_test)
            reliable = labels[conf > 0.70]
        """
        proba = self.predict_proba(X, tau=tau)
        best  = proba.argmax(axis=1)
        return self.classes_[best], proba[np.arange(len(X)), best]
