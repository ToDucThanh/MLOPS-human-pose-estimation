from abc import (
    ABC,
    abstractmethod,
)

from torch.utils.data import DataLoader

from config import DictDotNotation


class TrainerBase(ABC):
    def __init__(self, cfg: DictDotNotation):
        self.cfg = cfg

    @abstractmethod
    def prepare_data(self, train_ratio: float, *args, **kwargs) -> None:
        """
        Abstract method to prepare data for training.

        This method should be implemented in subclasses to prepare training and validation data.

        Parameters:
            train_ratio (float): Ratio of data to be used for training (e.g., 0.8 for 80% training data).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """
        Abstract property to get the training data loader.

        Subclasses should implement this property to provide a DataLoader for training data.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """
        Abstract property to get the validation data loader.

        Subclasses should implement this property to provide a DataLoader for validation data.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs) -> float:
        """
        Abstract method to train the model.

        Subclasses should implement this method to define the training loop.
        """
        raise NotImplementedError

    @abstractmethod
    def valid(self, *args, **kwargs) -> float:
        """
        Abstract method to valid the model.

        Subclasses should implement this method to define the validation loop.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Abstract method to fit the model.

        Subclasses should implement this method to define the end-to-end training process.
        """
        raise NotImplementedError
