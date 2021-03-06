import typing

from mid_term.exercise_3.src.classification.base_classifier import BaseClassifier
from mid_term.exercise_3.src.classification.postprocessing import combinatorial_truthify, compute_all_responses


class CombinatorialTruthifier(BaseClassifier):
    def __init__(self, classifier: BaseClassifier):
        self.classifier = classifier
        self.truthified_responses = {}

    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        self.classifier.fit(x, y)
        responses = compute_all_responses(self.classifier)
        self.truthified_responses = combinatorial_truthify(responses)

    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        outputs = []
        for sample in x:
            outputs.append(self.truthified_responses[tuple(sample)])
        return outputs

    def __str__(self):
        return f"CombinatorialTruthifier<(\n" \
               f"classifier={self.classifier}\n" \
               f")>"