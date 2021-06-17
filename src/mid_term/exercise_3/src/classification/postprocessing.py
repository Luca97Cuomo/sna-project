import typing

from mid_term.exercise_3.src.classification.base_classifier import BaseClassifier
from mid_term.exercise_3.src.dataset import generate_all_combinations
from mid_term.exercise_3.src.evaluation.accuracy import mae

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


def max_truthify(responses: ResponsesTable, desired_reviews_subject_to_cheating_index: float = 0.0) -> ResponsesTable:
    # calcolare quante disimbrogliare
    from mid_term.exercise_3.src.evaluation.truthfulness import cheatable_reviews, reviews_subject_to_cheating_from_responses

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


def combinatorial_truthify(responses: ResponsesTable) -> ResponsesTable:
    truthified_responses = {}

    # Calcolare tutte le combinazioni non considerando collegamenti incrociati tra i figli
    for i in range(1, 6 + 1): # 1, *, *, ..., 6, *, *
        # x, *, * = 3
        # there is only a combination where all samples are equale to 3.
        # evaluate mae
        min_combination = build_max_combination(root_value=i)
        min_metric = evaluate_combination(responses, min_combination, mae)

        # x, *, * = 2
        current_combination = {(i, None, None): 2}
        for j in range(1, 6 + 1):
            # x, x, *
            # x, * ,x
            # check childern score = 3
            subtree_min_metric, subtree_min_combination = evaluate_min_combination_score_three(i, j, responses, mae)

            # check children score = 2
            subtree_current_metric, subtree_current_combination = evaluate_min_combination_score_two(i, j, responses, mae)
            subtree_min_metric, subtree_min_combination = update_min_combination(subtree_min_metric, subtree_min_combination,
                                                                                 subtree_current_metric, subtree_current_combination)

            # non posso fare semplicemente l'update in quanto gli insiemi non hanno intersezione pari a 0
            # dovrei controllare se i valori associati sono diversi e in caso affermativo devo modificarli per renderli truthful
            # ad esempio aumentando il valore del sample non truthful.
            current_combination.update(subtree_min_combination)
        current_metric = evaluate_combination(responses, current_combination, mae)
        min_metric, min_combination = update_min_combination(min_metric, min_combination, current_metric, current_combination)

        # x, *, * = 1
        current_combination = {(i, None, None): 1}
        for j in range(1, 6 + 1):
            # check childern score = 3
            subtree_min_metric, subtree_min_combination = evaluate_min_combination_score_three(i, j, responses, mae)

            # check children score = 2
            subtree_current_metric, subtree_current_combination = evaluate_min_combination_score_two(i, j, responses, mae)
            subtree_min_metric, subtree_min_combination = update_min_combination(subtree_min_metric, subtree_min_combination,
                                                                                 subtree_current_metric, subtree_current_combination)

            # check children score = 1
            subtree_current_metric, subtree_current_combination = evaluate_min_combination_score_one(i, j, responses, mae)
            subtree_min_metric, subtree_min_combination = update_min_combination(subtree_min_metric, subtree_min_combination,
                                                                                 subtree_current_metric, subtree_current_combination)

            # non posso fare semplicemente l'update in quanto gli insiemi non hanno intersezione pari a 0
            # dovrei controllare se i valori associati sono diversi e in caso affermativo devo modificarli per renderli truthful
            # ad esempio aumentando il valore del sample non truthful.
            current_combination.update(subtree_min_combination)
        current_metric = evaluate_combination(responses, current_combination, mae)
        min_metric, min_combination = update_min_combination(min_metric, min_combination, current_metric, current_combination)

        # add current tree to full truthified responses dict
        truthified_responses.update(min_combination)

    return truthified_responses


def update_min_combination(min_metric, min_combination, current_metric, current_combination):
    if current_metric < min_metric:
        min_metric = current_metric
        min_combination = current_combination

    return min_metric, min_combination


def evaluate_min_combination_score_three(first_feature: int, second_feature: int, responses: ResponsesTable, metric_function: typing.Callable):
    # x, x, *
    # x, *, x
    combination = {(first_feature, second_feature, None): 3, (first_feature, None, second_feature): 3}

    for i in range(1, 6 + 1):
        combination[(first_feature, second_feature, i)] = 3
        combination[(first_feature, i, second_feature)] = 3

    return evaluate_combination(responses, combination, metric_function), combination


def evaluate_min_combination_score_two(first_feature: int, second_feature: int, responses: ResponsesTable, metric_function: typing.Callable):
    # x, x, *
    # x, *, x
    combination = {(first_feature, second_feature, None): 2, (first_feature, None, second_feature): 2}

    for i in range(1, 6 + 1):
        review = (first_feature, second_feature, i)
        if responses[review] < 2:
            combination[review] = 2
        else:
            combination[review] = responses[review]

        review = (first_feature, i, second_feature)
        if responses[review] < 2:
            combination[review] = 2
        else:
            combination[review] = responses[review]

    return evaluate_combination(responses, combination, metric_function), combination


def evaluate_min_combination_score_one(first_feature: int, second_feature: int, responses: ResponsesTable, metric_function: typing.Callable):
    # x, x, *
    # x, *, x
    combination = {(first_feature, second_feature, None): 1, (first_feature, None, second_feature): 1}

    for i in range(1, 6 + 1):
        review = (first_feature, second_feature, i)
        combination[review] = responses[review]

        review = (first_feature, i, second_feature)
        combination[review] = responses[review]

    return evaluate_combination(responses, combination, metric_function), combination


def build_max_combination(root_value: int):
    combination = {(root_value, None, None): 3}

    for i in range(1, 6 + 1):
        combination[(root_value, i, None)] = 3
        combination[(root_value, None, i)] = 3
        for j in range(1, 6 + 1):
            combination[(root_value, i, j)] = 3

    return combination # 49 samples


def evaluate_combination(responses: ResponsesTable, combination: ResponsesTable, metric: typing.Callable):
    # responses is bigger than combination, so we have to take only the keys that are in combination
    true_labels = []
    predicted_labels = []

    for review, score in combination.items():
        predicted_labels.append(score)
        true_labels.append(responses[review])

    return metric(predicted_labels, true_labels)
