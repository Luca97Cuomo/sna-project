import math
from unittest import TestCase
from evaluation.accuracy import mae, categorical_accuracy, mae_std, errors_greater_than_one


class TestAccuracy(TestCase):
    def setUp(self) -> None:
        self.predicted_labels = [2, 1, 3, 1, 1]
        self.actual_labels = [2, 1, 1, 3, 3]

    def test_mae(self):
        expected_mae = 6 / 5
        actual_mae = mae(self.predicted_labels, self.actual_labels)
        self.assertEqual(expected_mae, actual_mae)

    def test_mae_std(self):
        # mae = 6/5
        # (2*(6/5)^2 + 3*(4/5)^2) / 5
        # (72/25 + 48/25) / 5 = 120/25 / 5 = 120/125

        expected_mae_std = math.sqrt(120 / 125)
        actual_mae_std = mae_std(self.predicted_labels, self.actual_labels)
        self.assertAlmostEqual(expected_mae_std, actual_mae_std)

    def test_errors_greater_than_one(self):
        expected = 3 / 5
        actual = errors_greater_than_one(self.predicted_labels, self.actual_labels)
        self.assertEqual(expected, actual)

    def test_categorical_accuracy(self):
        expected_accuracy = 2 / 5
        actual_accuracy = categorical_accuracy(self.predicted_labels, self.actual_labels)
        self.assertEqual(expected_accuracy, actual_accuracy)