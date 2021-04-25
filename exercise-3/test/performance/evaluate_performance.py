import typing

from classification.logistic_regression import LogisticRegression
from dataset import generate_random_samples, generate_naive_labels_with_misreporting, generate_naive_labels, \
    generate_labels_using_only_available_features
from evaluation.accuracy import evaluate_over_all_combinations, mae
from evaluation.timing import evaluate_fit_time, evaluate_prediction_time
from evaluation.truthfulness import reviews_subject_to_cheating, ways_of_cheating_reviews
import logging
import numpy as np


def set_logging():
    logging.basicConfig(level=logging.INFO)


def print_statistics(classifiers_results: dict) -> None:
    for classifier_name, classifier_results in classifiers_results.items():
        print(f'\n##### {classifier_name} #####')
        for label_function_name, label_function_results in classifier_results.items():
            print(f'##### {label_function_name.upper()} #####')

            reviews_subject_to_cheating_values = label_function_results['reviews_subject_to_cheating_values']
            ways_of_cheating_reviews_values = label_function_results['ways_of_cheating_reviews_values']

            reviews_subject_to_cheating_mean = np.mean(reviews_subject_to_cheating_values)
            reviews_subject_to_cheating_std = np.std(reviews_subject_to_cheating_values)

            ways_of_cheating_reviews_mean = np.mean(ways_of_cheating_reviews_values)
            ways_of_cheating_reviews_std = np.std(ways_of_cheating_reviews_values)

            print('##### TRUTHFULNESS #####')
            print('reviews subject to cheating:')
            print(f'\tmean: {reviews_subject_to_cheating_mean}, std: {reviews_subject_to_cheating_std}')
            print('ways of cheating reviews:')
            print(f'\tmean: {ways_of_cheating_reviews_mean}, std: {ways_of_cheating_reviews_std}')

            print('##### METRICS #####')
            metrics_values = label_function_results['metrics_values']
            for metric_name, metric_results in metrics_values.items():
                print(f'metric name: {metric_name}')
                metric_over_all_combinations = metric_results['metric_over_all_combinations']
                metric_over_test_set = metric_results['metric_over_test_set']

                metric_over_all_combinations_mean = np.mean(metric_over_all_combinations)
                metric_over_all_combinations_std = np.std(metric_over_all_combinations)

                metric_over_test_set_mean = np.mean(metric_over_test_set)
                metric_over_test_set_std = np.std(metric_over_test_set)

                print('over all combinations:')
                print(f'\tmean: {metric_over_all_combinations_mean}, std: {metric_over_all_combinations_std}')
                print('over test set:')
                print(f'\tmean: {metric_over_test_set_mean}, std: {metric_over_test_set_std}\n')


def evaluate_all():
    """
    Evaluate performance for all the implemented classifiers
    for all the generate_labels functions
    for all the metrics
    for multiple training e test set
    """

    number_of_random_datasets = 10
    label_functions = [generate_naive_labels, generate_naive_labels_with_misreporting, generate_labels_using_only_available_features]
    classifier_classes = [LogisticRegression]
    metrics = [mae]

    classifiers_results = {classifier_class.__name__: {
        label_function.__name__: {
            'reviews_subject_to_cheating_values': [],
            'ways_of_cheating_reviews_values': [],
            'metrics_values': {}
        } for label_function in label_functions
    } for classifier_class in classifier_classes}

    for i in range(number_of_random_datasets):
        x_training = generate_random_samples(100, 0.2, 0.2)
        x_test = generate_random_samples(1000, 0.2, 0.2)

        for label_function in label_functions:
            y_training = label_function(x_training)
            y_test = label_function(x_test)

            for classifier_class in classifier_classes:
                classifier = classifier_class()

                classifier.fit(x_training, y_training)

                classifier_results = classifiers_results[classifier_class.__name__][label_function.__name__]

                reviews_subject_to_cheating_list = classifier_results['reviews_subject_to_cheating_values']
                ways_of_cheating_reviews_list = classifier_results['ways_of_cheating_reviews_values']

                reviews_subject_to_cheating_list.append(reviews_subject_to_cheating(classifier))
                ways_of_cheating_reviews_list.append(ways_of_cheating_reviews(classifier))

                y_predicted = classifier.predict(x_test)
                for metric in metrics:
                    metrics_dict = classifier_results['metrics_values']
                    metric_dict = metrics_dict.setdefault(metric.__name__, {})
                    metric_over_all_combinations_values = metric_dict.setdefault('metric_over_all_combinations', [])
                    metric_over_test_set_values = metric_dict.setdefault('metric_over_test_set', [])

                    metric_over_all_combinations_values.append(evaluate_over_all_combinations(classifier, label_function, metric))
                    metric_over_test_set_values.append(metric(y_predicted, y_test))

    print_statistics(classifiers_results)

    # f'fit_time: {fit_time[1] / fit_time[0]} s\n'
    # f'prediction_time: {prediction_time[1] / prediction_time[0]} s')


if __name__ == '__main__':
    set_logging()
    evaluate_all()