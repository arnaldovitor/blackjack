from typing import Any

from decouple import Csv, config
from loguru import logger

_AVAILABLE_METRICS: dict = {'accuracy': None}


def get_metrics() -> Any:
    """Returns the selected evaluation metrics based on the configuration.

    Returns:
        The selected evaluation metrics.
    """
    selected_metrics = config('METRICS', cast=Csv(str))
    for metric in selected_metrics:
        try:
            _ = _AVAILABLE_METRICS[metric]
        except Exception:
            logger.error(f'Metric: {metric} not available.')

    return config('METRICS', cast=Csv(str))
