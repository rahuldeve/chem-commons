from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from optuna.trial import FrozenTrial
from optuna.study import Study


class ObjectiveDirections(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveStrategy(ABC):
    @abstractmethod
    def __call__(self, study: Study) -> FrozenTrial:
        pass

    @abstractmethod
    def get_metrics(self) -> list[str]:
        pass

    @abstractmethod
    def get_directions(self) -> list[str]:
        pass


@dataclass
class SingleObjective(ObjectiveStrategy):
    metric: str
    direction: ObjectiveDirections

    def __call__(self, study: Study) -> FrozenTrial:
        return study.best_trial

    def get_metrics(self) -> list[str]:
        return [self.metric]

    def get_directions(self) -> list[str]:
        return [self.direction.value]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.metric}, {self.direction.value})"


@dataclass
class MultiObjectiveSum(ObjectiveStrategy):
    metrics: list[str]
    directions: list[ObjectiveDirections]

    def __call__(self, study: Study) -> FrozenTrial:
        best_trials = study.best_trials
        return max(best_trials, key=lambda x: sum(x.values))

    def get_metrics(self) -> list[str]:
        return self.metrics

    def get_directions(self) -> list[str]:
        return [d.value for d in self.directions]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({list(zip(self.metrics, self.directions))})"


@dataclass
class MultiObjectiveSingleBest(ObjectiveStrategy):
    metrics: list[str]
    selection_metric: str
    directions: list[ObjectiveDirections]

    def __call__(self, study: Study) -> FrozenTrial:
        best_trials = study.best_trials
        selection_metric_idx = self.metrics.index(self.selection_metric)
        metric_direction = self.directions[selection_metric_idx]

        if metric_direction == ObjectiveDirections.MAXIMIZE:
            return max(best_trials, key=lambda x: x.values[selection_metric_idx])
        elif metric_direction == ObjectiveDirections.MINIMIZE:
            return min(best_trials, key=lambda x: x.values[selection_metric_idx])
        else:
            raise Exception("Bad Objective Direction")

    def get_metrics(self) -> list[str]:
        return self.metrics

    def get_directions(self) -> list[str]:
        return list(map(lambda x: x.value, self.directions))

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        directions = [d.name for d in self.directions]
        metrics_str = str(list(zip(self.metrics, directions)))
        return f"{class_name}({metrics_str}, {self.selection_metric})"
