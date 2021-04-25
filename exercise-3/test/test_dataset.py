from unittest import TestCase
from dataset import generate_naive_labels, apply_naive_strategy, generate_all_combinations


class TestDataset(TestCase):
    def setUp(self) -> None:
        self.samples = [[1, 2, 3], [6, 1, 4], [5, None, 3], [5, 5, 4], [4, None, 1]]

    def test_generate_naive_labels(self):
        actual = generate_naive_labels(self.samples)
        expected = [1, 2, 2, 3, 2]

        self.assertListEqual(actual, expected)

    def test_apply_naive_strategy(self):
        apply_naive_strategy(self.samples)
        actual = self.samples
        expected = [[1, None, 3], [6, None, 4], [5, None, 3], [5, 5, 4], [4, None, None]]

        self.assertListEqual(actual, expected)

    def test_generate_all_combinations_total(self):
        expected_total = 294
        combinations = generate_all_combinations()
        self.assertEqual(len(combinations), expected_total)

    def test_generate_all_combinations_valid(self):
        expected = [[3, 1, 5], [1, None, None], [6, 6, None], [3, None, 2], [2, 2, 2]]
        combinations = generate_all_combinations()

        for expected_sample in expected:
            self.assertIn(expected_sample, combinations)

    def test_generate_all_combinations_invalid(self):
        expected = [[7, 1, 5], [None, None, None], [7, 6, None], [-2, None, 2], [0, 2, 2]]
        combinations = generate_all_combinations()

        for expected_sample in expected:
            self.assertNotIn(expected_sample, combinations)