from attrs import define
from tensorflow.keras import Sequential
from tensorflow.keras.aplications import MobileNetV2
from tensorflow.layers import Dense, GlobalAveragePooling2D


@define
class MobileNet:
    """A custom wrapper around MobileNetV2 for image classification.

    Attributes:
        num_classes: Number of classes for the classification problem.
        input_shape: Shape of the input images in the format (height, width, channels).
    """

    num_classes: int
    input_shape: tuple

    _base_model: MobileNetV2 = MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    _global_average_layer: GlobalAveragePooling2D = GlobalAveragePooling2D()
    _prediction_layer: Dense = Dense(num_classes, activation='softmax')

    model: Sequential = Sequential([_base_model, _global_average_layer, _prediction_layer])
