from unittest import TestCase
from src.dataset import generate_naive_labels, apply_naive_strategy


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
