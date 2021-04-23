import random
import typing


"""
Samples are characterised by 3 features:
- the first one, Food, is always an integer between 1 and 6, inclusive;
- the second and the third one, respectively Service and Value, are also integers in the 1 and 6 range, but may also be
not present (either because the restaurant cannot be evaluated on that feature or due to intentional misreporting). 
This characteristic is represented by the None Python value.

In the original problem, values were in the 0 and 5 range. It was shifted by 1 in order to possibly better disambiguate
between the non-existent value and 0.
"""


def _get_random_review_value(random_state: random.Random):
    return random_state.randint(1, 6)


def generate_random_samples(n_samples: int, service_null_percentage: float, value_null_percentage: float, seed: int = 42) \
        -> typing.Iterable[typing.List]:
    random_state = random.Random(seed)
    samples = []

    for i in range(n_samples):
        samples.append([])
        samples[i].append(_get_random_review_value(random_state))

        if random_state.random() < service_null_percentage:
            samples[i].append(None)
        else:
            samples[i].append(_get_random_review_value(random_state))

        if random_state.random() < value_null_percentage:
            samples[i].append(None)
        else:
            samples[i].append(_get_random_review_value(random_state))

    return samples


def generate_naive_labels(samples: typing.Iterable[typing.List], seed: int = 42) -> typing.List[int]:
    """

    """
    labels = []

    for sample in samples:
        score = 0
        for feature in sample:
            if feature is None:
                feature = 3
            score += feature

        if 1 <= score <= 6:
            score = 1
        elif 7 <= score <= 12:
            score = 2
        else:
            score = 3
        labels.append(score)

    return labels
