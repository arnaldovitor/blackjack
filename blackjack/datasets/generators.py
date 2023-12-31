from decouple import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from blackjack.utils.parsers import parse_target_size


def get_complex_generator(split_path: str) -> ImageDataGenerator:
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=config('SHEAR_RANGE', cast=float, default=0.0),
        zoom_range=config('ZOOM_RANGE', cast=float, default=0.0),
        rotation_range=config('ROTATION_RANGE', cast=int, default=0),
        width_shift_range=config('WIDTH_SHIFT_RANGE', cast=float, default=0.0),
        height_shift_range=config('HEIGHT_SHIFT_RANGE', cast=float, default=0.0),
        horizontal_flip=config('HORIZONTAL_FLIP', cast=bool, default=False),
    )

    generator = datagen.flow_from_directory(
        split_path,
        target_size=parse_target_size(config('TARGET_SIZE', default='224,224')),
        batch_size=config('BATCH_SIZE', cast=int, default=32),
        class_mode=config('CLASS_MODE', cast=str, default='categorical'),
    )

    return generator


def get_simple_generator(split_path: str) -> ImageDataGenerator:
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    generator = datagen.flow_from_directory(
        split_path,
        target_size=parse_target_size(config('TARGET_SIZE', cast=tuple, default='224, 224')),
        batch_size=config('BATCH_SIZE', cast=int, default=32),
        class_mode=config('CLASS_MODE', cast=str, default='categorical'),
    )

    return generator
