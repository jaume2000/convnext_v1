from pathlib import Path

import matplotlib.pyplot as plt

from metrics.metric import Metric


class MetricDeltaRatio(Metric):
    """Tracks ||delta|| / ||W|| for every part of every delta block over an epoch.

    The values come from the model's delta_ratios(), keyed "b{block}/{part}", and the
    epoch value of each key is the mean over the sampled steps. Three levels are
    exposed: the raw per-part ratios, their mean per block, and the mean over blocks,
    which is the scalar the rest of the history machinery (CSV, tqdm, print_last) uses.

    Reading the ratios costs a device sync, and they move on the scale of an epoch
    rather than of a step, so only every sample_every-th step is measured.
    """

    def __init__(self, sample_every: int = 20):
        super().__init__()
        self.sample_every = sample_every
        self.steps = 0
        self.total_ratios: dict[str, float] = {}
        self.last_mean: float | None = None

    def accumulate(self, model=None, **kwargs):
        if model is None or not hasattr(model, "delta_ratios"):
            return
        step = self.steps
        self.steps += 1
        if step % self.sample_every != 0:
            return
        ratios = model.delta_ratios()
        if not ratios:
            return
        for name, value in ratios.items():
            self.total_ratios[name] = self.total_ratios.get(name, 0.0) + float(value)
        self.total_samples += 1
        # Every block contributes the same number of parts, so the flat mean already is
        # the mean of the per-block means that compute_metric() reports.
        self.last_mean = sum(ratios.values()) / len(ratios)

    def pbar_value(self) -> str | None:
        if self.last_mean is None:
            return None
        return f"{self.last_mean:.2e}"

    def compute_parts(self) -> dict[str, float]:
        if self.total_samples == 0:
            return {}
        return {name: total / self.total_samples for name, total in self.total_ratios.items()}

    def compute_blocks(self) -> dict[str, float]:
        """Mean ratio per block, i.e. the per-part ratios averaged over the 5 parts."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, value in self.compute_parts().items():
            block = name.split("/")[0]
            totals[block] = totals.get(block, 0.0) + value
            counts[block] = counts.get(block, 0) + 1
        return {block: total / counts[block] for block, total in totals.items()}

    def compute_metric(self) -> float:
        blocks = self.compute_blocks()
        if not blocks:
            return 0.0
        return sum(blocks.values()) / len(blocks)

    def compute_components(self) -> dict[str, float]:
        parts = self.compute_parts()
        blocks = {f"{block}/mean": value for block, value in self.compute_blocks().items()}
        return {**parts, **blocks}

    @staticmethod
    def plot_series(history: list["MetricDeltaRatio"], plots_path: Path, name: str):
        """One figure per aggregation level: 45 parts, 9 blocks, 1 mean."""
        levels = [
            ("per_part", "per part", [metric.compute_parts() for metric in history], 5),
            ("per_block", "per block", [metric.compute_blocks() for metric in history], 1),
            ("mean", "mean", [
                {"mean": metric.compute_metric()} if metric.total_samples else {}
                for metric in history
            ], 0),
        ]
        for suffix, title, points, legend_columns in levels:
            MetricDeltaRatio._plot(
                MetricDeltaRatio._series(points),
                plots_path / f"{name}_{suffix}.png",
                f"{name}: delta ratio {title}",
                legend_columns,
            )

    @staticmethod
    def _series(points: list[dict[str, float]]) -> dict[str, list[float]]:
        """Transpose per-epoch dicts into one line per key.

        Epochs that never saw the key (a resumed run whose earlier epochs predate this
        metric) get a NaN so the line breaks instead of dropping to zero.
        """
        keys: list[str] = []
        for point in points:
            for key in point:
                if key not in keys:
                    keys.append(key)
        return {key: [point.get(key, float("nan")) for point in points] for key in keys}

    @staticmethod
    def _plot(series: dict[str, list[float]], path: Path, title: str, legend_columns: int):
        if not series:
            return
        fig, ax = plt.subplots(figsize=(11, 6) if legend_columns else (8, 5))
        for key, values in series.items():
            ax.plot(values, label=key, linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("||delta|| / ||W||")
        ax.set_title(title)
        if legend_columns:
            # Outside the axes: 45 labels inside would cover the curves they describe.
            ax.legend(
                fontsize=6,
                ncol=legend_columns,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
