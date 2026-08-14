import csv
from pathlib import Path
import pickle
from metrics.metric import Metric
import torch
import matplotlib.pyplot as plt

class Metrichistory:
    def __init__(self, path: Path, metric_cls: type[Metric], name: str):
        self.metricHistory: list[Metric] = []
        self.path = path
        self.metric_cls = metric_cls
        self.name = name
        self.best_metric = None
        self.best_metric_index = None
    def append(self, metric: Metric):
        self.metricHistory.append(metric)

    def compute_last(self):
        return self.metricHistory[-1].compute_metric()

    def create_point(self):
        metric = self.metric_cls()
        self.append(metric)
        self.save()

    def accumulate_last_point(self, pred: torch.Tensor, y_labels: torch.Tensor, loss: torch.Tensor):
        self.metricHistory[-1].accumulate(pred, y_labels, loss)

    def save(self):
        Path(self.path / "history").mkdir(parents=True, exist_ok=True)
        Path(self.path / "results").mkdir(parents=True, exist_ok=True)
        with open(self.path / "history" / f"{self.name}_history.pkl", "wb") as f:
            pickle.dump(self.metricHistory, f)
        self.toCSV()
    
    def toCSV(self):
        with open(self.path / "results" / f"{self.name}_history.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", self.metric_cls.__name__, "Loss"])
            for i, metric in enumerate(self.metricHistory):
                writer.writerow([i, metric.compute_metric(), metric.compute_loss()])

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

    def restore_best(self):
        if not self.metricHistory:
            self.best_metric = None
            self.best_metric_index = None
            return
        best_i = 0
        best_v = self.metricHistory[0].compute_metric()
        for i, metric in enumerate(self.metricHistory):
            value = metric.compute_metric()
            if value > best_v:
                best_v = value
                best_i = i
        self.best_metric = best_v
        self.best_metric_index = best_i

    def plot_history(self):
        Path(self.path / "plots").mkdir(parents=True, exist_ok=True)
        plt.plot([metric.compute_metric() for metric in self.metricHistory])
        plt.xlabel("Epoch")
        plt.ylabel(self.metric_cls.__name__)
        plt.title(f"{self.name} History")
        plt.savefig(self.path / "plots" / f"{self.name}_{self.metric_cls.__name__}_history.png")
        plt.close()

    def plot_history_loss(self):
        Path(self.path / "plots").mkdir(parents=True, exist_ok=True)
        plt.plot([metric.compute_loss() for metric in self.metricHistory])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.name} Loss History")
        plt.savefig(self.path / "plots" / f"{self.name}_loss_history.png")
        plt.close()

    def last_is_best(self):
        if self.best_metric is None:
            self.best_metric = self.metricHistory[0].compute_metric()
            self.best_metric_index = 0
        if self.metricHistory[-1].compute_metric() > self.best_metric:
            self.best_metric = self.metricHistory[-1].compute_metric()
            self.best_metric_index = len(self.metricHistory) - 1
        return self.best_metric_index == len(self.metricHistory) - 1

    def print_last(self):
        print(f"{self.name} {self.metric_cls.__name__}: {self.metricHistory[-1].compute_metric()}, Loss: {self.metricHistory[-1].compute_loss()}")