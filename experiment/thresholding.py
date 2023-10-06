from enum import Enum, auto
from ghostml import optimize_threshold_from_predictions
import numpy as np
from functools import partial
from typing import Callable
from numpy.typing import NDArray


def ghost_kappa_thresholding(y_true: NDArray, y_pred_prob: NDArray) -> float:
    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    optimal_threshold = optimize_threshold_from_predictions(
        y_true, y_pred_prob, thresholds, ThOpt_metrics="Kappa"
    )
    return optimal_threshold


def ghost_roc_thresholding(y_true: NDArray, y_pred_prob: NDArray):
    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    optimal_threshold = optimize_threshold_from_predictions(
        y_true, y_pred_prob, thresholds, ThOpt_metrics="ROC"
    )
    return optimal_threshold


def na_strategy(y_true: NDArray, y_pred_prob: NDArray):
    return 0.5


ThresholdFunction = Callable[[NDArray, NDArray], float]


class ThresholdStrategy(Enum):
    NA = auto()
    G_KAPPA = auto()
    G_ROC = auto()

    def function(self) -> ThresholdFunction:
        match self:
            case ThresholdStrategy.NA:
                return na_strategy
            case ThresholdStrategy.G_KAPPA:
                return ghost_kappa_thresholding
            case ThresholdStrategy.G_ROC:
                return ghost_roc_thresholding
