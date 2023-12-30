from abc import ABC, abstractmethod


class DatasetInterface(ABC):
    @abstractmethod
    def get_train(self) -> None:
        pass

    @abstractmethod
    def get_validation(self) -> None:
        pass

    @abstractmethod
    def get_test(self) -> None:
        pass
