from attrs import define, field
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from blackjack.datasets.dataset_interface import DatasetInterface
from blackjack.datasets.generators import get_complex_generator, get_simple_generator


@define
class BaseDataset(DatasetInterface):
    """
    Class for creating an instance of a basic classification dataset.

    The dataset should be organized as follows:
    - /path/to/dataset
        -- /path/to/dataset/train
            --- /path/to/dataset/train/class_1
            --- /path/to/dataset/train/class_2
        -- /path/to/dataset/val
            ...
        -- /path/to/dataset/test

    """

    train_path: str
    validation_path: str
    test_path: str

    _train_generator: ImageDataGenerator = field(init=False)
    _validation_genetator: ImageDataGenerator = field(init=False)
    _test_generator: ImageDataGenerator = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._train_generator = get_complex_generator(self.train_path)
        self._validation_generator = get_simple_generator(self.validation_path)
        self._test_generator = get_simple_generator(self.test_path)

    def get_train(self) -> ImageDataGenerator:
        """Get the training data generator.

        Returns:
            The generator for training data.
        """
        return self._train_generator

    def get_validation(self) -> ImageDataGenerator:
        """Get the validation data generator.

        Returns:
            The generator for validation data.
        """
        return self._validation_generator

    def get_test(self) -> ImageDataGenerator:
        """Get the test data generator.

        Returns:
            The generator for test data.
        """
        return self._test_generator
