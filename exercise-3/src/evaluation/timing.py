from classifiers.base_classifier import BaseClassifier
import timeit

from dataset import generate_all_combinations, generate_naive_labels


def evaluate_fit_time(classifier: BaseClassifier) -> float:
    samples = generate_all_combinations()
    labels = generate_naive_labels(samples)

    timer = timeit.Timer(lambda: classifier.fit(samples, labels))

    return timer.autorange()


def evaluate_prediction_time(fitted_classifier: BaseClassifier) -> float:
    samples = generate_all_combinations()

    timer = timeit.Timer(lambda: fitted_classifier.predict(samples))

    return timer.autorange()
