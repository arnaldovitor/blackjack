from typing import Any

from decouple import Csv, config


def build_metrics() -> Any:
    return config('METRICS', cast=Csv(str))
