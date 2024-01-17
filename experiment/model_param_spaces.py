from dataclasses import dataclass
from typing import Optional, Any, TypeAlias


@dataclass
class FloatParam:
    low: float
    high: float
    step: Optional[float] = None
    log: Optional[bool] = None


@dataclass
class IntParam:
    low: int
    high: int
    step: int = 1


@dataclass
class ChoiceParam:
    choices: list


@dataclass
class ConstantParam:
    constant: Any


ParamType: TypeAlias = FloatParam | IntParam | ChoiceParam | ConstantParam
ParamSpace: TypeAlias = dict[str, ParamType]

lgbm_classifier_param_space: ParamSpace = {
    "lgbmclassifier__objective": ConstantParam("binary"),
    "lgbmclassifier__metric": ConstantParam("average_precision"),
    "lgbmclassifier__verbosity": ConstantParam(-1),
    "lgbmclassifier__boosting_type": ChoiceParam(["gbdt", "dart"]),
    "lgbmclassifier__reg_alpha": FloatParam(1e-8, 1e-1, log=True),
    "lgbmclassifier__reg_lambda": FloatParam(1e-8, 1e-1, log=True),
    "lgbmclassifier__num_leaves": IntParam(2, 256),
    "lgbmclassifier__subsample": FloatParam(0.1, 1),
    "lgbmclassifier__colsample_bytree": FloatParam(0.1, 1),
    "lgbmclassifier__min_child_samples": IntParam(5, 100),
    "lgbmclassifier__n_jobs": ConstantParam(16),
    "lgbmclassifier__random_state": ConstantParam(42),
    "lgbmclassifier__scale_pos_weight": IntParam(30, 100, 1),
    "lgbmclassifier__n_estimators": ConstantParam(2000),
    "lgbmclassifier__max_depth": IntParam(5, 50)
    # "lgbmclassifier__early_stopping_rounds": ConstantParam(100)
}

xgboost_param_space: ParamSpace = {
    "xgbclassifier__n_estimators": ConstantParam(2000),
    # "xgbclassifier__early_stopping_rounds": ConstantParam(100),
    "xgbclassifier__max_depth": IntParam(5, 10),
    "xgbclassifier__min_child_weight": FloatParam(0.1, 1),
    "xgbclassifier__learning_rate": FloatParam(0.001, 0.01),
    "xgbclassifier__subsample": FloatParam(0.1, 1),
    "xgbclassifier__n_jobs": ConstantParam(4),
    "xgbclassifier__random_state": ConstantParam(42),
    "xgbclassifier__scale_pos_weight": IntParam(10, 50, 1),
    "xgbclassifier__verbosity": ConstantParam(0),
    # "xgbclassifier__eval_metric": ChoiceParam(['aucpr', 'auc'])
}

catboost_param_space: ParamSpace = {
    "catboostclassifier__verbose": ConstantParam(False),
    "catboostclassifier__allow_writing_files": ConstantParam(False),
    "catboostclassifier__iterations": IntParam(100, 500),
    "catboostclassifier__depth": IntParam(1, 10),
    "catboostclassifier__learning_rate": FloatParam(0.001, 0.1),
    "catboostclassifier__l2_leaf_reg": FloatParam(0.001, 0.1),
    "catboostclassifier__border_count": IntParam(32, 255),
    "catboostclassifier__bagging_temperature": FloatParam(0, 1),
    # "catboostclassifier__random_strength": IntParam(0, 1),
    "catboostclassifier__random_state": ConstantParam(42),
}
