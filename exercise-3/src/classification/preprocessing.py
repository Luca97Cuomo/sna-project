import typing


def map_missing_features_to_zero(samples: typing.Iterable[typing.List]) -> None:
    for sample in samples:
        for i in range(len(sample)):
            if sample[i] is None:
                sample[i] = 0
