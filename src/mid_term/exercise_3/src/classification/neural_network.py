import typing
from copy import deepcopy

from sklearn.neural_network import MLPClassifier

from mid_term.exercise_3.src.classification.base_classifier import BaseClassifier
from mid_term.exercise_3.src.classification.preprocessing import map_missing_features_to_zero


class NeuralNetwork(BaseClassifier):
    def __init__(self, hidden_layer_sizes=(30, 30, 20, 10), regularization=False):
        self.regularization = regularization
        self.hidden_layer_sizes = hidden_layer_sizes

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=1e-4 if self.regularization else 0,
        )

    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        samples = deepcopy(x)
        map_missing_features_to_zero(samples)

        self.model.fit(samples, y)

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        samples = deepcopy(x)
        map_missing_features_to_zero(samples)

        return self.model.predict(samples)

    def __str__(self):
        return f"NeuralNetwork<(\n" \
               f"   regularization={self.regularization}\n" \
               f"   hidden_layers={self.hidden_layer_sizes}\n" \
               f")>"
