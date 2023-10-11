from enum import Enum, auto
from ghostml import optimize_threshold_from_predictions
import numpy as np
from typing import Callable
from numpy.typing import ArrayLike


def ghost_kappa_thresholding(y_true: ArrayLike, y_pred_prob: ArrayLike) -> float:
    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    optimal_threshold = optimize_threshold_from_predictions(
        y_true, y_pred_prob, thresholds, ThOpt_metrics="Kappa"
    )
    return optimal_threshold


def ghost_roc_thresholding(y_true: ArrayLike, y_pred_prob: ArrayLike):
    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    optimal_threshold = optimize_threshold_from_predictions(
        y_true, y_pred_prob, thresholds, ThOpt_metrics="ROC"
    )
    return optimal_threshold


def na_strategy(y_true: ArrayLike, y_pred_prob: ArrayLike):
    return 0.5


ThresholdFunction = Callable[[ArrayLike, ArrayLike], float]


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
