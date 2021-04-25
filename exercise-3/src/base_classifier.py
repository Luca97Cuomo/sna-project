from abc import abstractmethod
import typing


class BaseClassifier:
    @abstractmethod
    def fit(self, x: typing.List[typing.List], y: typing.List[int]) -> None:
        raise NotImplementedError("Must be implemented by subclasses.")

    @abstractmethod
    def predict(self, x: typing.List[typing.List]) -> typing.List[int]:
        raise NotImplementedError("Must be implemented by subclasses.")
