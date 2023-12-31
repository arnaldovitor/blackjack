from typing import Any

from attrs import define, field
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


@define
class MobileNet:
    """A custom wrapper around MobileNetV2 for image classification.

    Attributes:
        num_classes: Number of classes for the classification problem.
        input_shape: Shape of the input images in the format (height, width, channels).
    """

    num_classes: int
    input_shape: tuple

    _base_model: Any = field(init=False)
    _global_average_layer: Any = field(init=False)
    _prediction_layer: Any = field(init=False)

    model: Any = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._base_model: MobileNetV2 = MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights='imagenet'
        )
        self._global_average_layer: GlobalAveragePooling2D = GlobalAveragePooling2D()
        self._prediction_layer: Dense = Dense(self.num_classes, activation='softmax')

        self.model: Sequential = Sequential(
            [self._base_model, self._global_average_layer, self._prediction_layer]
        )
