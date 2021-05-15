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


def truthify(responses: ResponsesTable, desired_reviews_subject_to_cheating_index: float = 0.0) -> ResponsesTable:
    # calcolare quante disimbrogliare
    from evaluation.truthfulness import cheatable_reviews, reviews_subject_to_cheating_from_responses

    current_reviews_subject_to_cheating_index = reviews_subject_to_cheating_from_responses(responses)
    if desired_reviews_subject_to_cheating_index >= current_reviews_subject_to_cheating_index:
        return responses

    reviews_to_truthify = int((current_reviews_subject_to_cheating_index - desired_reviews_subject_to_cheating_index) * cheatable_reviews)
    number_of_truthified_reviews = 0

    truthified_responses = {}

    for input, output in responses.items():
        if number_of_truthified_reviews < reviews_to_truthify:
            new_score = output

            if input[1] is not None and input[2] is not None:
                new_score = max(responses[(input[0], None, input[2])], new_score)
                new_score = max(responses[(input[0], input[1], None)], new_score)
                new_score = max(responses[(input[0], None, None)], new_score)
            elif input[1] is not None:
                new_score = max(responses[(input[0], None, input[2])], new_score)
            elif input[2] is not None:
                new_score = max(responses[(input[0], input[1], None)], new_score)

            if new_score > output:
                number_of_truthified_reviews += 1
        else:
            new_score = output

        truthified_responses[input] = new_score

    return truthified_responses
