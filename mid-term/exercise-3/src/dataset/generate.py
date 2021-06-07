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
        -> typing.List[typing.List]:
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


def generate_naive_labels_with_misreporting(samples: typing.Iterable[typing.List]) -> typing.List[int]:
    """
    It creates labels as a linear function of the sum of the features, but considering a None value as a score of 3,
    the middle ground. This is purposefully done in order to make a misreporting strategy viable.
    """
    labels = []

    for sample in samples:
        score = 0
        for feature in sample:
            if feature is None:
                feature = 3
            score += feature

        score = feature_sum_to_score(score)
        labels.append(score)

    return labels


def feature_sum_to_score(feature_sum, num_available_features=3):
    if num_available_features == 3:
        first_endpoint = 1
        second_endpoint = 6
        third_endpoint = 12
        fourth_endpoint = 18
    elif num_available_features == 2:
        first_endpoint = 1
        second_endpoint = 4
        third_endpoint = 8
        fourth_endpoint = 12
    elif num_available_features == 1:
        first_endpoint = 1
        second_endpoint = 2
        third_endpoint = 4
        fourth_endpoint = 6
    else:
        raise ValueError('num_available_features must be in [1, 2, 3]')

    if first_endpoint <= feature_sum <= second_endpoint:
        return 1
    elif second_endpoint < feature_sum <= third_endpoint:
        return 2
    elif third_endpoint < feature_sum <= fourth_endpoint:
        return 3
    else:
        raise ValueError(f'feature sum must be in [{first_endpoint}, {fourth_endpoint}]')


def generate_labels_using_only_available_features(samples: typing.Iterable[typing.List]) -> typing.List[int]:
    labels = []

    for sample in samples:
        score = 0
        available_features = 3
        for feature in sample:
            if feature is None:
                available_features -= 1
                continue
            score += feature

        score = feature_sum_to_score(score, num_available_features=available_features)
        labels.append(score)

    return labels


def generate_naive_labels(samples: typing.Iterable[typing.List]) -> typing.List[int]:
    labels = []

    for sample in samples:
        score = 0
        for feature in sample:
            if feature is None:
                feature = 0
            score += feature

        score = feature_sum_to_score(score)
        labels.append(score)

    return labels


def generate_all_combinations() -> typing.List[typing.List]:
    samples = []

    for food in range(1, 6 + 1):
        for service in range(0, 6 + 1):
            for value in range(0, 6 + 1):
                samples.append([food, service if service > 0 else None, value if value > 0 else None])

    return samples
