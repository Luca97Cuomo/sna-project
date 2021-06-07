from unittest import TestCase
from dataset import generate_naive_labels_with_misreporting, apply_naive_strategy, generate_all_combinations, \
    generate_naive_labels, feature_sum_to_score, generate_labels_using_only_available_features


class TestDataset(TestCase):
    def setUp(self) -> None:
        self.samples = [[1, 2, 3], [6, 1, 4], [5, None, 3], [5, 5, 4], [4, None, 1], [5, None, None]]

    def test_generate_naive_labels_with_misreporting(self):
        actual = generate_naive_labels_with_misreporting(self.samples)
        expected = [1, 2, 2, 3, 2, 2]

        self.assertListEqual(actual, expected)

    def test_generate_labels_using_only_available_features(self):
        actual = generate_labels_using_only_available_features(self.samples)
        expected = [1, 2, 2, 3, 2, 3]

        self.assertListEqual(actual, expected)

    def test_generate_naive_labels(self):
        actual = generate_naive_labels(self.samples)
        expected = [1, 2, 2, 3, 1, 1]

        self.assertListEqual(actual, expected)

    def test_apply_naive_strategy(self):
        apply_naive_strategy(self.samples)
        actual = self.samples
        expected = [[1, None, 3], [6, None, 4], [5, None, 3], [5, 5, 4], [4, None, None], [5, None, None]]

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

    def test_feature_sum_to_score_3_features(self):
        num_features = 3
        actual_score = feature_sum_to_score(1, num_available_features=num_features)
        self.assertEqual(1, actual_score)
        actual_score = feature_sum_to_score(6, num_available_features=num_features)
        self.assertEqual(1, actual_score)

        actual_score = feature_sum_to_score(7, num_available_features=num_features)
        self.assertEqual(2, actual_score)
        actual_score = feature_sum_to_score(12, num_available_features=num_features)
        self.assertEqual(2, actual_score)

        actual_score = feature_sum_to_score(13, num_available_features=num_features)
        self.assertEqual(3, actual_score)
        actual_score = feature_sum_to_score(18, num_available_features=num_features)
        self.assertEqual(3, actual_score)

        with self.assertRaises(ValueError):
            feature_sum_to_score(0, num_available_features=num_features)
        with self.assertRaises(ValueError):
            feature_sum_to_score(19, num_available_features=num_features)

    def test_feature_sum_to_score_2_features(self):
        num_features = 2
        actual_score = feature_sum_to_score(1, num_available_features=num_features)
        self.assertEqual(1, actual_score)
        actual_score = feature_sum_to_score(4, num_available_features=num_features)
        self.assertEqual(1, actual_score)

        actual_score = feature_sum_to_score(5, num_available_features=num_features)
        self.assertEqual(2, actual_score)
        actual_score = feature_sum_to_score(8, num_available_features=num_features)
        self.assertEqual(2, actual_score)

        actual_score = feature_sum_to_score(9, num_available_features=num_features)
        self.assertEqual(3, actual_score)
        actual_score = feature_sum_to_score(12, num_available_features=num_features)
        self.assertEqual(3, actual_score)

        with self.assertRaises(ValueError):
            feature_sum_to_score(0, num_available_features=num_features)
        with self.assertRaises(ValueError):
            feature_sum_to_score(13, num_available_features=num_features)

    def test_feature_sum_to_score_1_features(self):
        num_features = 1
        actual_score = feature_sum_to_score(1, num_available_features=num_features)
        self.assertEqual(1, actual_score)
        actual_score = feature_sum_to_score(2, num_available_features=num_features)
        self.assertEqual(1, actual_score)

        actual_score = feature_sum_to_score(3, num_available_features=num_features)
        self.assertEqual(2, actual_score)
        actual_score = feature_sum_to_score(4, num_available_features=num_features)
        self.assertEqual(2, actual_score)

        actual_score = feature_sum_to_score(5, num_available_features=num_features)
        self.assertEqual(3, actual_score)
        actual_score = feature_sum_to_score(6, num_available_features=num_features)
        self.assertEqual(3, actual_score)

        with self.assertRaises(ValueError):
            feature_sum_to_score(0, num_available_features=num_features)
        with self.assertRaises(ValueError):
            feature_sum_to_score(7, num_available_features=num_features)
