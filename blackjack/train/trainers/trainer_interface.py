from abc import ABC, abstractmethod


class TrainerInterface(ABC):
    @abstractmethod
    def _compile(self) -> None:
        pass

    @abstractmethod
    def _fit(self) -> None:
        pass

    @abstractmethod
    def start_train(self) -> None:
        pass
