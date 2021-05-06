import typing

from classification.base_classifier import BaseClassifier
from classification.postprocessing import truthify, compute_all_responses


class Truthifier(BaseClassifier):
    def __init__(self, classifier: BaseClassifier, desired_truthfulness_index: float = 0.0):
        self.classifier = classifier
        self.desired_truthfulness_index = desired_truthfulness_index
        self.truthified_responses = {}

    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        self.classifier.fit(x, y)
        responses = compute_all_responses(self.classifier)
        self.truthified_responses = truthify(
            responses,
            desired_reviews_subject_to_cheating_index=self.desired_truthfulness_index
        )

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        outputs = []
        for sample in x:
            outputs.append(self.truthified_responses[tuple(sample)])
        return outputs

    def __str__(self):
        return f"Truthifier<(\n" \
               f"classifier={self.classifier}\n" \
               f"desired_truthfulness_index={self.desired_truthfulness_index}\n" \
               f")>"
