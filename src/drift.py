"""
Embedding drift: are users asking about different things than our data covers?

Reference = the average (centroid) embedding of documents in Qdrant.
Recent    = the average embedding of the last `window` user queries.
drift     = 1 - cosine(recent, reference)

If users start asking about topics far from the indexed data, drift rises,
which usually means retrieval quality is about to drop. Prometheus alerts on it.
"""
from collections import deque

import numpy as np

from src import metrics


class DriftMonitor:

    def __init__(self, reference=None, window=200, min_samples=20):
        self.reference = None
        self.recent = deque(maxlen=window)
        self.min_samples = min_samples
        if reference is not None:
            self.set_reference(reference)

    def set_reference(self, vectors):
        centroid = np.mean(np.asarray(vectors), axis=0)
        self.reference = centroid / np.linalg.norm(centroid)

    def update(self, vector):
        self.recent.append(np.asarray(vector))

        if self.reference is None or len(self.recent) < self.min_samples:
            return None

        centroid = np.mean(self.recent, axis=0)
        score = float(1 - centroid @ self.reference / np.linalg.norm(centroid))
        metrics.QUERY_DRIFT.set(score)
        return score
