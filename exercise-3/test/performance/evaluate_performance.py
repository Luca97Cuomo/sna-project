import datetime
import sys
import typing
from pathlib import Path

from classification.base_classifier import BaseClassifier
from classification.combinatorial_truthifier import CombinatorialTruthifier
from classification.logistic_regression import LogisticRegression
from classification.linear_regression import LinearRegression
from classification.neural_network import NeuralNetwork
from classification.thrutifier import Truthifier
from dataset import generate_random_samples, generate_naive_labels_with_misreporting, generate_naive_labels, \
    generate_labels_using_only_available_features
from evaluation.accuracy import evaluate_over_all_combinations, mae, categorical_accuracy, mae_std, \
    errors_greater_than_one
from evaluation.timing import evaluate_fit_time, evaluate_prediction_time
from evaluation.truthfulness import reviews_subject_to_cheating, ways_of_cheating_reviews
import logging
import logging.config
import numpy as np


def set_logging():
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    logging.config.dictConfig({
        "version": 1,
        "formatters": {
            "named": {
                "format": "%(name)s - %(message)s"
            },
            "unnamed": {
                "format": "%(message)s"
            }
        },
        "handlers": {
            "console-named": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "named"
            },
            "console-unnamed": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "unnamed"
            },
            "file-named": {
                "class": "logging.FileHandler",
                "filename": results_dir / f"results_{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt",
                "formatter": "named"
            },
            "file-unnamed": {
                "class": "logging.FileHandler",
                "filename": results_dir / f"results_{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt",
                "formatter": "unnamed"
            },
        },
        "loggers": {
            "root": {
                "level": "INFO",
                "handlers": ["console-unnamed", "file-unnamed"]
            },
            "classification": {
                "level": "DEBUG",
                "handlers": ["console-named", "file-named"],
                "propagate": False
            }
        },
    })


def print_statistics(classifiers_results: dict, descriptions: dict) -> None:
    for classifier_class, classifier_results in classifiers_results.items():
        logging.info(f'\n##### CLASSIFIER #####\n'
                     f'{descriptions[classifier_class]}\n'
                     f'#####\n')
        for label_function_name, label_function_results in classifier_results.items():
            logging.info(f'##### {label_function_name.upper()} #####')

            reviews_subject_to_cheating_values = label_function_results['reviews_subject_to_cheating_values']
            ways_of_cheating_reviews_values = label_function_results['ways_of_cheating_reviews_values']

            reviews_subject_to_cheating_mean = np.mean(reviews_subject_to_cheating_values)
            reviews_subject_to_cheating_std = np.std(reviews_subject_to_cheating_values)

            ways_of_cheating_reviews_mean = np.mean(ways_of_cheating_reviews_values)
            ways_of_cheating_reviews_std = np.std(ways_of_cheating_reviews_values)

            logging.info('##### TRUTHFULNESS #####')
            logging.info('reviews subject to cheating:')
            logging.info(f'\tmean: {reviews_subject_to_cheating_mean}, std: {reviews_subject_to_cheating_std}')
            logging.info('ways of cheating reviews:')
            logging.info(f'\tmean: {ways_of_cheating_reviews_mean}, std: {ways_of_cheating_reviews_std}')

            logging.info('##### METRICS #####')
            metrics_values = label_function_results['metrics_values']
            for metric_name, metric_results in metrics_values.items():
                logging.info(f'metric name: {metric_name}')
                metric_over_all_combinations = metric_results['metric_over_all_combinations']
                metric_over_test_set = metric_results['metric_over_test_set']

                metric_over_all_combinations_mean = np.mean(metric_over_all_combinations)
                metric_over_all_combinations_std = np.std(metric_over_all_combinations)

                metric_over_test_set_mean = np.mean(metric_over_test_set)
                metric_over_test_set_std = np.std(metric_over_test_set)

                logging.info('over all combinations:')
                logging.info(f'\tmean: {metric_over_all_combinations_mean}, std: {metric_over_all_combinations_std}')
                logging.info('over test set:')
                logging.info(f'\tmean: {metric_over_test_set_mean}, std: {metric_over_test_set_std}\n')


def evaluate_all():
    """
    Evaluate performance for all the implemented classifiers
    for all the generate_labels functions
    for all the metrics
    for multiple training e test set
    """

    number_of_random_datasets = 1
    label_functions = [generate_naive_labels, generate_naive_labels_with_misreporting, generate_labels_using_only_available_features]
    classifier_classes = [
        LogisticRegression,
        #lambda: LinearRegression(positive=True),
        # NeuralNetwork,
        #lambda: Truthifier(NeuralNetwork(), desired_truthfulness_index=0.0),
        #lambda: Truthifier(NeuralNetwork(), desired_truthfulness_index=0.1),
        #lambda: Truthifier(NeuralNetwork(), desired_truthfulness_index=0.05),
        #lambda: CombinatorialTruthifier(NeuralNetwork())
        # lambda: Truthifier(NeuralNetwork(), desired_truthfulness_index=1.0)
    ]
    metrics = [mae, errors_greater_than_one, categorical_accuracy]

    classifiers_results = {classifier_class: {
        label_function.__name__: {
            'reviews_subject_to_cheating_values': [],
            'ways_of_cheating_reviews_values': [],
            'metrics_values': {}
        } for label_function in label_functions
    } for classifier_class in classifier_classes}

    descriptions = {}

    for i in range(number_of_random_datasets):
        x_training = generate_random_samples(1000, 0.2, 0.2)
        x_test = generate_random_samples(1000, 0.2, 0.2)

        for label_function in label_functions:
            y_training = label_function(x_training)
            y_test = label_function(x_test)

            for classifier_class in classifier_classes:
                classifier = classifier_class()

                classifier.fit(x_training, y_training)

                descriptions[classifier_class] = str(classifier)
                classifier_results = classifiers_results[classifier_class][label_function.__name__]

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

    print_statistics(classifiers_results, descriptions)

    # f'fit_time: {fit_time[1] / fit_time[0]} s\n'
    # f'prediction_time: {prediction_time[1] / prediction_time[0]} s')


def test_classifier_simple(classifier: BaseClassifier):
    x_training = generate_random_samples(1000, 0.2, 0.2)
    x_test = generate_random_samples(1000, 0.2, 0.2)

    label_function = generate_naive_labels

    classifier.fit(x_training, label_function(x_training))
    predictions = classifier.predict(x_test)

    mae_score = mae(predictions, label_function(x_test))
    cheated_reviews = reviews_subject_to_cheating(classifier)

    print(f"mae: {mae_score}")
    print(f"cheated reviews: {cheated_reviews}")


if __name__ == '__main__':
    set_logging()
    evaluate_all()
    # classifier = NeuralNetwork()
    # test_classifier_simple(classifier)
