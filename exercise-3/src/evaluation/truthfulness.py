"""
A fit classifier has its truthfulness tested by evaluating all possible reviews.
The chosen measures for the truthfulness of a classifier are two:

- the reviews subject to cheating: the ratio between the number of samples for which hiding one or more
    features results in a better score and the total number of samples that have feature to be hidden (for instance,
    1 x x is not included).


- the ways of cheating: the ratio between the total number of misreportings that result in a better score and the
    total possible number of misreportings. Therefore, if a review admits different ways to cheat the final score,
    this measure takes it into account, giving an idea of how "easy" it can be for a strategy to cheat.

Consider the following example (incomplete since the measures apply to the whole space of possible reviews, whereas
the example considers a space of only 2 reviews in order to stay simple):

    Review1: 1 1 2
    Misreportings with better score:
    1 x 2
    1 x x
    1 1 x

    Review2: 3 4 4
    Misreportings with better score:
    3 x 4

The first measure would evaluate to 1.0, because a strategy can cheat on both reviews.
The second measure would instead evaluate to 4/6 = 0.66, because the best strategy is limited in the ways it can cheat
the second review.
"""
import typing

from classification.base_classifier import BaseClassifier
from classification.postprocessing import compute_all_responses, ResponsesTable
from dataset.generate import generate_all_combinations


total_samples = 6 * 7 * 7

samples_with_no_missing_features = 6 * 6 * 6
samples_with_two_missing_features = 6 * 1 * 1
samples_with_one_missing_feature = (6 * 1 * 6) + (6 * 6 * 1)

possible_misreportings = samples_with_no_missing_features * 3 + samples_with_one_missing_feature
cheatable_reviews = total_samples - samples_with_two_missing_features


def cheatable(review: tuple, score: int, responses: ResponsesTable,
              index_to_cheat: int) -> bool:
    if index_to_cheat == 1:
        cheating_sample = (review[0], None, review[2])
    elif index_to_cheat == 2:
        cheating_sample = (review[0], review[1], None)
    else:
        raise ValueError("Index to cheat must be in [1, 2]")
    cheating_score = responses[cheating_sample]
    return cheating_score > score


def reviews_subject_to_cheating(fitted_classifier: BaseClassifier) -> float:
    responses = compute_all_responses(fitted_classifier)

    return reviews_subject_to_cheating_from_responses(responses)


def reviews_subject_to_cheating_from_responses(responses):
    cheated_reviews = 0
    for sample, score in responses.items():
        # Since it is unknown how the classifier decides, two missing values may be less favorable than
        # one missing value, so all the possible cheating ways are checked
        if sample[1] is not None and cheatable(sample, score, responses, 1):
            cheated_reviews += 1
            continue
        if sample[2] is not None and cheatable(sample, score, responses, 2):
            cheated_reviews += 1
            continue
        if sample[1] is not None and sample[2] is not None:
            sample = (sample[0], None, sample[2])
            if cheatable(sample, score, responses, 2):
                cheated_reviews += 1
    return cheated_reviews / cheatable_reviews


def ways_of_cheating_reviews(fitted_classifier: BaseClassifier) -> float:
    responses = compute_all_responses(fitted_classifier)

    misreportings = 0

    for sample, score in responses.items():
        # Since it is unknown how the classifier decides, two missing values may be less favorable than
        # one missing value, so all the possible cheating ways are checked
        if sample[1] is not None and cheatable(sample, score, responses, 1):
            misreportings += 1
        if sample[2] is not None and cheatable(sample, score, responses, 2):
            misreportings += 1
        if sample[1] is not None and sample[2] is not None:
            sample = (sample[0], None, sample[2])
            if cheatable(sample, score, responses, 2):
                misreportings += 1

    return misreportings / possible_misreportings
