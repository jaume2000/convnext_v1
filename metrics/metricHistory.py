import csv
from typing import Dict
from pathlib import Path
import pickle
from metrics.metric import Metric
import matplotlib.pyplot as plt


class DictHistoryMetrics:
    def __init__(self, path: Path, split: str):
        self.path = path
        self.split = split
        self.histories: Dict[str, MetricHistory] = {}

    def addHistoryMetric(self, name: str, metricClass: type[Metric], higher_is_better: bool = True):
        full_name = f"{self.split}_{name}"
        self.histories[name] = MetricHistory(
            self.path, metricClass, full_name, higher_is_better=higher_is_better
        )

    def accumulate(self, **kwargs):
        for histmetric in self.histories.values():
            histmetric.accumulate_last_point(**kwargs)

    def create_point(self):
        for histmetric in self.histories.values():
            histmetric.create_point()

    def get(self, name: str) -> "MetricHistory":
        return self.histories[name]

    def num_epochs(self) -> int:
        if not self.histories:
            return 0
        return min(len(h.metricHistory) for h in self.histories.values())

    def plot_history(self, names: list[str] | None = None):
        names = names if names is not None else list(self.histories.keys())
        Path(self.path / "plots").mkdir(parents=True, exist_ok=True)
        for name in names:
            self.histories[name].plot_history()

    def save(self):
        for histmetric in self.histories.values():
            histmetric.save()

    def load(self) -> bool:
        if not self.histories:
            return False
        return all(histmetric.load() for histmetric in self.histories.values())

    def drop_empty_trailing(self):
        for histmetric in self.histories.values():
            histmetric.drop_empty_trailing()

    def truncate(self, n: int):
        for histmetric in self.histories.values():
            histmetric.truncate(n)

    def restore_best(self, name: str | None = None):
        if name is not None:
            self.histories[name].restore_best()
            return
        for histmetric in self.histories.values():
            histmetric.restore_best()

    def last_is_best(self, name: str) -> bool:
        return self.histories[name].last_is_best()

    def print_last(self):
        for histmetric in self.histories.values():
            histmetric.print_last()


class MetricHistory:
    def __init__(
        self,
        path: Path,
        metric_cls: type[Metric],
        name: str,
        higher_is_better: bool = True,
    ):
        self.metricHistory: list[Metric] = []
        self.path = path
        self.metric_cls = metric_cls
        self.name = name
        self.higher_is_better = higher_is_better
        self.best_metric = None
        self.best_metric_index = None

    def append(self, metric: Metric):
        self.metricHistory.append(metric)

    def compute_last(self):
        return self.metricHistory[-1].compute_metric()

    def create_point(self):
        self.append(self.metric_cls())

    def accumulate_last_point(self, **kwargs):
        self.metricHistory[-1].accumulate(**kwargs)

    def save(self):
        Path(self.path / "history").mkdir(parents=True, exist_ok=True)
        Path(self.path / "results").mkdir(parents=True, exist_ok=True)
        with open(self.path / "history" / f"{self.name}_history.pkl", "wb") as f:
            pickle.dump(self.metricHistory, f)
        self.toCSV()

    def toCSV(self):
        with open(self.path / "results" / f"{self.name}_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "value"])
            for i, metric in enumerate(self.metricHistory):
                if metric.total_samples == 0:
                    continue
                writer.writerow([i, float(metric.compute_metric())])

    def history_path(self) -> Path:
        return self.path / "history" / f"{self.name}_history.pkl"

    def load(self) -> bool:
        path = self.history_path()
        if not path.is_file():
            return False
        with open(path, "rb") as f:
            self.metricHistory = pickle.load(f)
        return True

    def drop_empty_trailing(self):
        while self.metricHistory and self.metricHistory[-1].total_samples == 0:
            self.metricHistory.pop()

    def truncate(self, n: int):
        self.metricHistory = self.metricHistory[:n]

    def _is_better(self, value, best) -> bool:
        if self.higher_is_better:
            return value > best
        return value < best

    def restore_best(self):
        if not self.metricHistory:
            self.best_metric = None
            self.best_metric_index = None
            return
        best_i = 0
        best_v = self.metricHistory[0].compute_metric()
        for i, metric in enumerate(self.metricHistory):
            value = metric.compute_metric()
            if self._is_better(value, best_v):
                best_v = value
                best_i = i
        self.best_metric = best_v
        self.best_metric_index = best_i

    def plot_history(self):
        Path(self.path / "plots").mkdir(parents=True, exist_ok=True)
        values = [float(metric.compute_metric()) for metric in self.metricHistory]
        plt.plot(values)
        plt.xlabel("Epoch")
        plt.ylabel(self.metric_cls.__name__)
        plt.title(f"{self.name} History")
        plt.savefig(self.path / "plots" / f"{self.name}_{self.metric_cls.__name__}_history.png")
        plt.close()

    def last_is_best(self) -> bool:
        if not self.metricHistory:
            return False
        current = self.metricHistory[-1].compute_metric()
        if self.best_metric is None:
            self.best_metric = current
            self.best_metric_index = len(self.metricHistory) - 1
            return True
        if self._is_better(current, self.best_metric):
            self.best_metric = current
            self.best_metric_index = len(self.metricHistory) - 1
        return self.best_metric_index == len(self.metricHistory) - 1

    def print_last(self):
        print(f"{self.name}: {self.metricHistory[-1].compute_metric()}")
