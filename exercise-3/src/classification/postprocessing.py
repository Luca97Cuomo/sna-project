import typing

from classification.base_classifier import BaseClassifier
from dataset import generate_all_combinations


ResponsesTable = typing.Dict[
    typing.Tuple[int, typing.Optional[int], typing.Optional[int]],
    int]


def compute_all_responses(fitted_classifier: BaseClassifier) -> ResponsesTable:
    all_inputs = generate_all_combinations()
    all_predictions = fitted_classifier.predict(all_inputs)

    all_responses = {}

    for input, output in zip(all_inputs, all_predictions):
        all_responses[tuple(input)] = output

    return all_responses


def truthify(responses: ResponsesTable) -> ResponsesTable:
    truthified_responses = {}

    for input, output in responses.items():
        max_score = output

        if input[1] is not None and input[2] is not None:
            max_score = max(responses[(input[0], None, input[2])], max_score)
            max_score = max(responses[(input[0], input[1], None)], max_score)
            max_score = max(responses[(input[0], None, None)], max_score)
        elif input[1] is not None:
            max_score = max(responses[(input[0], None, input[2])], max_score)
        elif input[2] is not None:
            max_score = max(responses[(input[0], input[1], None)], max_score)

        truthified_responses[input] = max_score

    return truthified_responses

