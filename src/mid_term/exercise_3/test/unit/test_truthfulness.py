from unittest import TestCase

import typing

from mid_term.exercise_3.src.classification.base_classifier import BaseClassifier
from mid_term.exercise_3.src.evaluation.truthfulness import cheatable, reviews_subject_to_cheating, ways_of_cheating_reviews


class IdempotentMockClassifier(BaseClassifier):
    def __init__(self, returned_score: int):
        self.returned_score = returned_score

    def fit(self, x: typing.List[typing.Iterable], y: typing.Iterable[int]) -> None:
        pass

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        return [self.returned_score for _ in range(len(x))]


class MaxForNoneMockClassifier(BaseClassifier):
    def fit(self, x: typing.List[typing.Iterable], y: typing.Iterable[int]) -> None:
        pass

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        predictions = []

        for sample in x:
            if None in sample:
                predictions.append(3)
            else:
                predictions.append(1)

        return predictions


class IdempotentDict:
    def __init__(self, returned_score):
        self.returned_score = returned_score

    def __getitem__(self, item):
        return self.returned_score


class TestTruthfulness(TestCase):
    def test_cheatable(self):
        review = [2, 3, 5]
        response = IdempotentDict(returned_score=2)
        is_cheatable = cheatable(review=review, score=1, responses=response, index_to_cheat=1)
        self.assertTrue(is_cheatable)

    def test_not_cheatable(self):
        review = [2, 3, 5]
        response = IdempotentDict(returned_score=1)
        is_cheatable = cheatable(review=review, score=1, responses=response, index_to_cheat=1)
        self.assertFalse(is_cheatable)

    def test_reviews_subject_to_cheating_truthful(self):
        fitted_classifier = IdempotentMockClassifier(returned_score=42)
        truthful_index = reviews_subject_to_cheating(fitted_classifier)
        self.assertEqual(truthful_index, 0)

    def test_reviews_subject_to_cheating_untruthful(self):
        combinations = 294
        possible_cheatable_reviews = combinations - 6
        uncheatable_reviews = (6 * 1 * 6) + (6 * 6 * 1)  # all reviews with already None in them
        cheated_reviews = possible_cheatable_reviews - uncheatable_reviews
        expected_truthful_index = cheated_reviews / possible_cheatable_reviews

        fitted_classifier = MaxForNoneMockClassifier()
        actual_truthful_index = reviews_subject_to_cheating(fitted_classifier)
        self.assertEqual(expected_truthful_index, actual_truthful_index)

    def test_ways_of_cheating_reviews_truthful(self):
        fitted_classifier = IdempotentMockClassifier(returned_score=42)
        truthful_index = ways_of_cheating_reviews(fitted_classifier)
        self.assertEqual(truthful_index, 0)

    def test_ways_of_cheating_reviews_untruthful(self):
        samples_with_no_missing_features = 6 * 6 * 6
        samples_with_one_missing_feature = (6 * 1 * 6) + (6 * 6 * 1)
        possible_misreportings = samples_with_no_missing_features * 3 + samples_with_one_missing_feature

        # the used classifier cannot be fooled if a feature is None
        total_misreportings = samples_with_no_missing_features * 3

        expected_truthful_index = total_misreportings / possible_misreportings

        fitted_classifier = MaxForNoneMockClassifier()
        actual_truthful_index = ways_of_cheating_reviews(fitted_classifier)
        self.assertEqual(expected_truthful_index, actual_truthful_index)
