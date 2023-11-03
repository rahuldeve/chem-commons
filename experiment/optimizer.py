import warnings
from copy import deepcopy
from functools import partial

import numpy as np
import optuna
import ray
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.pipeline import Pipeline, make_pipeline

from .base import (
    ChoiceParam,
    ConstantParam,
    Experiment,
    FloatParam,
    IntParam,
    ParamSpace,
)
from .confidence import delong_confidence_intervals
from .utils import RayExperimentTracker, set_seeds
from .fit_params import fit_param_mapping


def sample_param_space(trial, param_space: ParamSpace):
    sampled_params = dict()
    for name, p in param_space.items():
        match p:
            case FloatParam(low, high, step, log):
                sampled_params[name] = trial.suggest_float(
                    name, low, high, step=step, log=log
                )
            case IntParam(low, high, step):
                sampled_params[name] = trial.suggest_int(name, low, high, step=step)
            case ChoiceParam(choices):
                sampled_params[name] = trial.suggest_categorical(name, choices)
            case ConstantParam(constant):
                trial.set_user_attr(name, constant)
                sampled_params[name] = constant
            case _:
                raise Exception("Bad Param Type", p)

    return sampled_params


def calc_scores(y_pred_prob, y_pred, y_true, split_name):
    metrics_dict = dict()
    metrics_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    metrics_dict["balanced_accuracy"] = metrics.balanced_accuracy_score(y_true, y_pred)
    metrics_dict["f1"] = metrics.f1_score(y_true, y_pred)
    metrics_dict["precision"] = metrics.precision_score(y_true, y_pred)
    metrics_dict["recall"] = metrics.recall_score(y_true, y_pred)
    metrics_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_pred_prob)
    metrics_dict["average_precision"] = metrics.average_precision_score(
        y_true, y_pred_prob
    )

    auc, (lb, ub) = delong_confidence_intervals(y_true, y_pred_prob)
    metrics_dict["test_delong_auc"] = auc
    metrics_dict["lb"] = lb
    metrics_dict["ub"] = ub

    return {f"{split_name}_{k}": v for k, v in metrics_dict.items()}


def objective(
    trial,
    pipeline,
    pipeline_param_space,
    X_train,
    y_train,
    X_val,
    y_val,
    objective_metrics,
    threshold_strategy,
    sample_weights=None,
):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seeds(42)

    params = sample_param_space(trial, pipeline_param_space)
    pipeline = pipeline.set_params(
        **params,
        # **{k:v for k,v in trial.user_attrs.items() if k!='cross_val_scores'}
    )

    # We need to pass X_val to eval_set but it does not get passed through all the pipeline steps
    # So we first extract the feature pipeline and transform X_val
    # Since we are fitting the feature pipeline part two times, there may be a mismatch between the steps
    # If columns are different, then an error should be thrown. Not sure about values within columns
    feature_pipeline = deepcopy(pipeline.steps[0][1])
    feature_pipeline = feature_pipeline.fit(X_train, y_train)
    transformed_X_val = feature_pipeline.transform(X_val)

    fit_params = fit_param_mapping[pipeline.steps[-1][1].__class__](transformed_X_val, y_val)
    fit_params = {f'{pipeline.steps[-1][0]}__{k}': v for k,v in fit_params.items()}

    # fit_params = {
    #     f'{pipeline.steps[-1][0]}__eval_set': [(transformed_X_val, y_val)], 
    #     # f'{pipeline.steps[-1][0]}__verbose': True,
    #     f'{pipeline.steps[-1][0]}__early_stopping_rounds': 100
    # }

    pipeline = pipeline.fit(X_train, y_train, **fit_params)
    

    y_train_pred_prob = pipeline.predict_proba(X_train)[:, 1]
    thresholding_func = threshold_strategy.function()
    optimal_threshold = thresholding_func(y_train, y_train_pred_prob)

    y_val_pred_prob = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = np.where(y_val_pred_prob >= optimal_threshold, 1.0, 0.0)

    scores = calc_scores(y_val_pred_prob, y_val_pred, y_val, "val")
    return [scores[f"val_{metric}"] for metric in objective_metrics]


@ray.remote(num_cpus=16)
def optimize_pipeline(experiment: Experiment, tracker: RayExperimentTracker):  # type: ignore
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seeds(42)

    feature_pipeline = experiment.feature_pipeline.pipeline
    feature_pipeline_param_space = experiment.feature_pipeline.param_space
    terminal_model_cls = experiment.terminal_model.value

    pipeline: Pipeline = make_pipeline(feature_pipeline, terminal_model_cls())
    pipeline_param_space = {
        **{
            f"{feature_pipeline.__class__.__name__.lower()}__{k}": v
            for k, v in feature_pipeline_param_space.items()
        },
        **experiment.terminal_model.get_param_space(),
    }

    X_train = experiment.data.X_train
    y_train = experiment.data.y_train
    X_val = experiment.data.X_val
    y_val = experiment.data.y_val
    X_test = experiment.data.X_test
    y_test = experiment.data.y_test

    study = optuna.create_study(
        sampler=experiment.sampler.optuna_sampler,
        directions=experiment.objective_strategy.get_directions(),
    )

    objective_func = partial(
        objective,
        pipeline=deepcopy(pipeline),
        pipeline_param_space=deepcopy(pipeline_param_space),
        X_train=deepcopy(X_train),
        y_train=deepcopy(y_train),
        X_val=deepcopy(X_val),
        y_val=deepcopy(y_val),
        objective_metrics=deepcopy(experiment.objective_strategy.get_metrics()),
        threshold_strategy=deepcopy(experiment.threshold_strategy)
        # sample_weights = experiment.sample_weights
    )

    study.optimize(
        objective_func,
        n_trials=experiment.n_trials,
        show_progress_bar=False,
        callbacks=[lambda study, trial: tracker.update.remote()],
    )

    best_trial = experiment.objective_strategy(study)
    pipeline.set_params(
        **best_trial.params,
        **{k: v for k, v in best_trial.user_attrs.items() if k != "cross_val_scores"},
    ).set_output(
        transform="pandas"
    )  # type: ignore

    feature_pipeline = deepcopy(pipeline.steps[0][1])
    feature_pipeline = feature_pipeline.fit(X_train, y_train)
    transformed_X_val = feature_pipeline.transform(X_val)
    fit_params = fit_param_mapping[pipeline.steps[-1][1].__class__](transformed_X_val, y_val)
    fit_params = {f'{pipeline.steps[-1][0]}__{k}': v for k,v in fit_params.items()}

    pipeline = pipeline.fit(X_train, y_train, **fit_params)
    y_train_pred_prob = pipeline.predict_proba(X_train)[:, 1]  # type: ignore
    thresholding_func = experiment.threshold_strategy.function()
    optimal_threshold = thresholding_func(y_train, y_train_pred_prob)

    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = np.where(y_test_pred_prob >= optimal_threshold, 1.0, 0.0)
    test_scores = calc_scores(y_test_pred_prob, y_test_pred, y_test.to_numpy(), "test")
    test_scores = {k: np.round(v, 3) for k, v in test_scores.items()}
    return test_scores, pipeline, optimal_threshold
