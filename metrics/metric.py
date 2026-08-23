class Metric:

    def __init__(self):
        self.total_samples = 0

    def accumulate(self, **kwargs) -> None:
        pass

    def compute_metric(self):
        pass

    def compute_components(self) -> dict:
        """Named sub-values behind compute_metric(), written as extra CSV columns."""
        return {}

    def pbar_value(self) -> str | None:
        """Formatted value for the progress bar, or None to stay out of it."""
        return None
