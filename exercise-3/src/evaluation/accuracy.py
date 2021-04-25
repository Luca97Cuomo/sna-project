import typing

from classification.base_classifier import BaseClassifier
from dataset import generate_all_combinations


def mae(predicted_labels: typing.List[int], true_labels: typing.List[int]) -> float:
    if len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels and true_labels have different length")

    if len(predicted_labels) == 0:
        raise ValueError("The length is 0")

    error_sum = 0
    for predicted, true in zip(predicted_labels, true_labels):
        error_sum += abs(predicted - true)

    return error_sum / len(predicted_labels)


def evaluate_over_all_combinations(fitted_classifier: BaseClassifier,
                                   generate_labels: typing.Callable[[typing.Iterable[typing.List]], typing.List[int]],
                                   evaluation_metric: typing.Callable[[typing.List[int], typing.List[int]], float]) -> float:
    samples = generate_all_combinations()
    true_labels = generate_labels(samples)

    predicted_labels = fitted_classifier.predict(samples)

    return evaluation_metric(predicted_labels, true_labels)
