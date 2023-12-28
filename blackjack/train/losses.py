from typing import Any

from decouple import config
from tensorflow.leras.losses import BinaryCrossentropy

_AVAILABLE_LOSSES: dict = {'bce': BinaryCrossentropy}


def build_loss() -> Any:
    selected_loss = _AVAILABLE_LOSSES[config('LOSS')]
    return selected_loss(from_logits=True)
