class Metric:

    def __init__(self):
        self.total_samples = 0

    def accumulate(self, **kwargs) -> None:
        pass

    def compute_metric(self):
        pass
