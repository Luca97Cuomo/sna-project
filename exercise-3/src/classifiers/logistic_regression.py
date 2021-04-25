import typing

from classifiers.base_classifier import BaseClassifier
import sklearn.linear_model


class LogisticRegression(BaseClassifier):
    def __init__(self):
        self.model = sklearn.linear_model.LogisticRegression(penalty='none', multi_class='multinomial')

    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        self.model.fit(x, y)

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        pass

