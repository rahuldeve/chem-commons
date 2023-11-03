from enum import Enum
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
import pandas as pd
from typing import Tuple


def xgboost_fit_params(X_val, y_val):
    return {"eval_set": [(X_val, y_val)], "verbose": False, "early_stopping_rounds": 50}


def lightgbm_fit_params(X_val, y_val):
    return {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "logloss",
        "callbacks": [
            early_stopping(stopping_rounds=100, verbose=False)
        ],
    }

def catboost_fit_params(X_val, y_val):
    return {
        "eval_set": [(X_val, y_val)],
        "verbose_eval": False,
        "use_best_model": True,
        "early_stopping_rounds": 50

    }


fit_param_mapping = {
    XGBClassifier: xgboost_fit_params,
    LGBMClassifier: lightgbm_fit_params,
    CatBoostClassifier: catboost_fit_params
}
