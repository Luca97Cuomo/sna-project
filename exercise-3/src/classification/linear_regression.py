import logging
import typing

from classification.base_classifier import BaseClassifier
import sklearn.linear_model
from copy import deepcopy

from classification.preprocessing import map_missing_features_to_zero

logger = logging.getLogger(__name__)


class LinearRegression(BaseClassifier):
    def __init__(self, positive: bool = False, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.model = sklearn.linear_model.LinearRegression(positive=positive, fit_intercept=fit_intercept)

    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        samples = deepcopy(x)
        map_missing_features_to_zero(samples)
        self.model.fit(samples, y)

        logger.debug('intercept: %s, coefficients: %s', self.model.intercept_, self.model.coef_)

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        samples = deepcopy(x)
        map_missing_features_to_zero(samples)
        return LinearRegression._predictions_postprocessing(self.model.predict(samples))

    @staticmethod
    def _predictions_postprocessing(predictions: typing.List[float]) -> typing.List[int]:
        def from_float_to_int(float_prediction: float) -> int:
            int_prediction = round(float_prediction)

            if int_prediction < 1:
                int_prediction = 1
            elif int_prediction > 3:
                int_prediction = 3

            return int_prediction

        return list(map(from_float_to_int, predictions))

    def __str__(self):
        return f"LinearRegression<(\n" \
               f"   regularization={False}\n" \
               f"   fit_intercept={self.fit_intercept}\n" \
               f"   positive={self.positive}\n" \
               f")>"
