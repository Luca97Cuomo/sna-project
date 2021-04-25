from unittest import TestCase
from evaluation.accuracy import mae


class TestAccuracy(TestCase):
    def test_mae(self):
        predicted_labels = [2, 1, 3, 1, 1]
        actual_labels = [2, 1, 1, 3, 3]
        expected_mae = 6 / 5
        actual_mae = mae(predicted_labels, actual_labels)
        self.assertEqual(expected_mae, actual_mae)
