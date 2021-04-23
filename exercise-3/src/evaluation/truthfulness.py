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

from src.base_classifier import BaseClassifier


def cheatable(review: typing.List[typing.Optional[int]], score: int, fitted_classifier: BaseClassifier,
              index_to_cheat: int) -> bool:
    cheating_sample = review[:]
    cheating_sample[index_to_cheat] = None
    cheating_score = fitted_classifier.predict([cheating_sample])[0]
    return cheating_score > score


def possible_samples() -> typing.List[typing.List]:
    samples = []

    for food in range(1, 6 + 1):
        for service in range(0, 6 + 1):
            for value in range(0, 6 + 1):
                samples.append([food, service if service > 0 else None, value if value > 0 else None])

    return samples


def reviews_subject_to_cheating(fitted_classifier: BaseClassifier) -> float:
    samples = possible_samples()

    cheated_reviews = 0
    cheatable_reviews = len(samples) - 6  # excluding 1 x x, 2 x x, ... which cannot be cheated

    for sample in samples:
        score = fitted_classifier.predict([sample])[0]
        # Since it is unknown how the classifier decides, two missing values may be less favorable than
        # one missing value, so all the possible cheating ways are checked
        if sample[1] is not None and cheatable(sample, score, fitted_classifier, 1):
            cheatable_reviews += 1
            continue
        if sample[2] is not None and cheatable(sample, score, fitted_classifier, 2):
            cheatable_reviews += 1
            continue
        if sample[1] is not None and sample[2] is not None:
            sample[1] = None
            if cheatable(sample, score, fitted_classifier, 2):
                cheatable_reviews += 1

    return cheated_reviews / cheatable_reviews
