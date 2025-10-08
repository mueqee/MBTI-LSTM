"""
MBTI-LSTM: Классификация типов личности на основе LSTM

Этот пакет предоставляет инструменты для обучения и использования
LSTM моделей для классификации типов личности MBTI на основе
текстов из социальных сетей.
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import training
from . import utils

__all__ = [
    "data",
    "models",
    "training",
    "utils",
]

