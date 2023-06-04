import unittest

import numpy as np

from knn import (
    euclidian_distance,
    chebyshev_distance,
    cosine_distance
)

class TestKNN(unittest.TestCase):

    def test_euclidean_distance(self):
        a = np.array([[1, 0, 0]])
        b = np.array([[-1, 0, 0]])
        d = euclidian_distance(a, b)

        self.assertAlmostEqual(d, 2)

    def test_chebyshev_distance(self):
        a = np.array([[1, 0, 0]])
        b = np.array([[-1, 0, 0]])
        d = chebyshev_distance(a, b)

        self.assertAlmostEqual(d, 2)