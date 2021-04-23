from abc import abstractmethod
import typing


class BaseClassifier:
    @abstractmethod
    def fit(self, x: typing.Iterable[typing.Iterable], y: typing.Iterable[int]) -> None:
        raise NotImplementedError("Must be implemented by subclasses.")

    @abstractmethod
    def predict(self, x: typing.Iterable[typing.Iterable]) -> typing.List[int]:
        raise NotImplementedError("Must be implemented by subclasses.")
