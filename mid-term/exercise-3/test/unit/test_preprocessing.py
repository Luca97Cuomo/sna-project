from unittest import TestCase

from classification.preprocessing import map_missing_features_to_zero


class TestPreprocessing(TestCase):
    def test_map_missing_features_to_zero(self):
        samples = [[2, None], [None, None], [3, 2], [None, 1]]
        expected = [[2, 0], [0, 0], [3, 2], [0, 1]]

        map_missing_features_to_zero(samples)

        self.assertListEqual(expected, samples)
