import typing
import random


"""
Strategies 
"""


def apply_naive_strategy(samples: typing.Iterable[typing.List]):
    for sample in samples:
        if sample[1] and sample[1] < 3:
            sample[1] = None
        if sample[2] and sample[2] < 3:
            sample[2] = None

