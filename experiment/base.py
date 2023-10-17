from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TypeAlias

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.samplers import BaseSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .model_param_spaces import *
from .objective_strategies import ObjectiveStrategy
from .thresholding import *


class SupportedModels(Enum):
    XGBClassifier = XGBClassifier
    LGBMClassifier = LGBMClassifier
    CatBoostClassifier = CatBoostClassifier

    def get_param_space(self) -> ParamSpace:
        model_param_space_mapping: dict[SupportedModels, ParamSpace] = {
            SupportedModels.XGBClassifier: xgboost_param_space,
            SupportedModels.LGBMClassifier: lgbm_classifier_param_space,
            SupportedModels.CatBoostClassifier: catboost_param_space,
        }

        return model_param_space_mapping[self]


@dataclass
class Results:
    test_scores: dict[str, float]
    optimized_pipeline: Pipeline
    optimal_threshold: int

    def to_dict(self) -> dict[str, Any]:
        return self.test_scores | {
            "optimized_pipeline": self.optimized_pipeline,
            "threshold": self.optimal_threshold,
        }


@dataclass
class DataBunch:
    name: Optional[str]
    X_train: pd.DataFrame
    y_train: pd.DataFrame | pd.Series
    X_val: pd.DataFrame
    y_val: pd.DataFrame | pd.Series
    X_test: pd.DataFrame
    y_test: pd.DataFrame | pd.Series

    def __str__(self) -> str:
        return str(self.name)


@dataclass
class FeaturePipeline:
    name: Optional[str]
    pipeline: Pipeline
    param_space: ParamSpace

    def __str__(self) -> str:
        return str(self.name)


@dataclass
class Sampler:
    name: str
    optuna_sampler: BaseSampler

    def __str__(self) -> str:
        return self.name


@dataclass
class Experiment:
    feature_pipeline: FeaturePipeline
    terminal_model: SupportedModels
    data: DataBunch
    objective_strategy: ObjectiveStrategy
    n_trials: int
    sampler: Sampler
    threshold_strategy: ThresholdStrategy
    # sample_weights: Optional[pd.Series] = None
    results: Optional[Results] = None

    def __str__(self) -> str:
        components = [
            str(self.feature_pipeline),
            str(self.objective_strategy),
            str(self.sampler),
            str(self.threshold_strategy),
        ]

        return " / ".join(components)

    def to_dict(self) -> dict[str, float]:
        results_dict = self.results.to_dict() if self.results else {}
        others = {
            "feature_pipeline": str(self.feature_pipeline),
            "dataset": str(self.data),
            "objective_strategy": str(self.objective_strategy),
            "threshold_strategy": str(self.threshold_strategy),
            "sampler_name": self.sampler,
        }
        return results_dict | others
