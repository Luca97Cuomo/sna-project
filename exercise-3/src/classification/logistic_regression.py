import logging
import typing

from classification.base_classifier import BaseClassifier
import sklearn.linear_model
from copy import deepcopy

from classification.preprocessing import map_missing_features_to_zero

logger = logging.getLogger(__name__)


class LogisticRegression(BaseClassifier):
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.model = sklearn.linear_model.LogisticRegression(penalty='none', multi_class='multinomial', fit_intercept=fit_intercept)

    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        samples = deepcopy(x)
        map_missing_features_to_zero(samples)
        self.model.fit(samples, y)

        logger.debug('intercept: %s, coefficients: %s', self.model.intercept_, self.model.coef_)

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        samples = deepcopy(x)
        map_missing_features_to_zero(samples)
        return self.model.predict(samples)

    def __str__(self):
        return f"LogisticRegression<(\n" \
               f"   regularization={False}\n" \
               f"   fit_intercept={self.fit_intercept}\n" \
               f")>"
