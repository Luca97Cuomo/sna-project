import math
import typing

from mid_term.exercise_3.src.classification.base_classifier import BaseClassifier
from mid_term.exercise_3.src.dataset import generate_all_combinations


def mae(predicted_labels: typing.List[int], true_labels: typing.List[int]) -> float:
    if len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels and true_labels have different length")

    if len(predicted_labels) == 0:
        raise ValueError("The length is 0")

    error_sum = 0
    for predicted, true in zip(predicted_labels, true_labels):
        error_sum += abs(predicted - true)

    return error_sum / len(predicted_labels)


def mae_std(predicted_labels: typing.List[int], true_labels: typing.List[int]) -> float:
    if len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels and true_labels have different length")

    if len(predicted_labels) == 0:
        raise ValueError("The length is 0")

    error_mean = mae(predicted_labels, true_labels)

    error_sum = 0
    for predicted, true in zip(predicted_labels, true_labels):
        error = abs(predicted - true)
        error_sum += (error - error_mean)**2

    return math.sqrt(error_sum / len(predicted_labels))


def categorical_accuracy(predicted_labels: typing.List[int], true_labels: typing.List[int]) -> float:
    if len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels and true_labels have different length")

    if len(predicted_labels) == 0:
        raise ValueError("The length is 0")

    sum = 0
    for predicted, true in zip(predicted_labels, true_labels):
        sum += int(predicted == true)

    return sum / len(predicted_labels)


def errors_greater_than_one(predicted_labels: typing.List[int], true_labels: typing.List[int]) -> float:
    if len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels and true_labels have different length")

    if len(predicted_labels) == 0:
        raise ValueError("The length is 0")

    sum = 0
    for predicted, true in zip(predicted_labels, true_labels):
        error = abs(predicted - true)
        if error > 1:
            sum += 1

    return sum / len(predicted_labels)


def evaluate_over_all_combinations(fitted_classifier: BaseClassifier,
                                   generate_labels: typing.Callable[[typing.Iterable[typing.List]], typing.List[int]],
                                   evaluation_metric: typing.Callable[[typing.List[int], typing.List[int]], float]) -> float:
    samples = generate_all_combinations()
    true_labels = generate_labels(samples)

    predicted_labels = fitted_classifier.predict(samples)

    return evaluation_metric(predicted_labels, true_labels)
