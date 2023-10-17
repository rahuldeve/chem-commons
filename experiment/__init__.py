from itertools import product
from copy import deepcopy

import ray
from tqdm.auto import tqdm

from .optimizer import optimize_pipeline
from .base import *
from .model_param_spaces import *
from .objective_strategies import *
from .thresholding import *
from .utils import RayExperimentTracker


def generate_experiments(
    feature_pipelines: list[FeaturePipeline],
    models: list[SupportedModels],
    samplers: list[Sampler],
    objective_strategies: list[ObjectiveStrategy],
    threshold_strategies: list[ThresholdStrategy],
    data: DataBunch,
    n_trials: int,
    sample_weights=None,
):
    experiments = []
    for entry in product(
        feature_pipelines, samplers, models, objective_strategies, threshold_strategies
    ):
        entry = deepcopy(entry)
        
        feature_pipeline = entry[0]
        sampler = entry[1]
        model = entry[2]
        objective_strategy = entry[3]
        threshold_strategy = entry[4]

        experiments.append(
            Experiment(
                feature_pipeline=feature_pipeline,
                terminal_model=model,
                data=data,
                objective_strategy=objective_strategy,
                sampler=sampler,
                threshold_strategy=threshold_strategy,
                n_trials=n_trials,
            )
        )

    return experiments


def optimize_all(experiments: list[Experiment]):
    out = []
    trackers = [RayExperimentTracker.remote() for _ in experiments]
    jobs = [optimize_pipeline.remote(e, t) for e,t in zip(experiments, trackers)] # type: ignore
    pbars = [tqdm(total=e.n_trials) for e in experiments]

    remaining = jobs
    while remaining:
        completed, remaining = ray.wait(remaining, timeout=5)

        for job in remaining:
            idx = jobs.index(job)
            pbar = pbars[idx]
            tracker = trackers[idx]
            pbar.n = ray.get(tracker.get_progress.remote()) # type: ignore
            pbar.refresh()
            pbars[idx] = pbar

        for job in completed:
            idx = jobs.index(job)
            pbars[idx].close()
            test_scores, pipeline, optimal_thershold = ray.get(job)
            experiment = experiments[idx]
            experiment.results = Results(test_scores, pipeline, optimal_thershold)
            out.append(experiment)

    # out = []
    # for experiment in experiments:
    #     test_scores, pipeline, _ = optimize_pipeline(experiment, None)
    #     experiment.results = Results(test_scores, pipeline)
    #     out.append(experiment)

    return pd.DataFrame.from_records([exp.to_dict() for exp in out])
