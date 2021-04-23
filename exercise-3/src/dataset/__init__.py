from .generate import *
from .strategy import *


def honest_dataset(n_samples: int, service_null_percentage: float, value_null_percentage: float, seed: int = 42):
    samples = generate_random_samples(n_samples, service_null_percentage, value_null_percentage, seed)
    labels = generate_naive_labels(samples, seed)
    return [samples, labels]


def naively_corrupt_dataset(n_samples: int, service_null_percentage: float, value_null_percentage: float, seed: int = 42):
    samples = generate_random_samples(n_samples, service_null_percentage, value_null_percentage, seed)
    apply_naive_strategy(samples)
    labels = generate_naive_labels(samples, seed)
    return [samples, labels]
