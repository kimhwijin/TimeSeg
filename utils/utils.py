import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


class EarlyStopping:

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        assert mode in ("min", "max"), "`mode` must be 'min' or 'max'"
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")
        self.num_bad    = 0
        self.stop       = False

    def __call__(self, metric_value: float) -> bool:
        improved = (
            metric_value < self.best_value - self.min_delta  
            if self.mode == "min"
            else metric_value > self.best_value + self.min_delta 
        )
        if improved:
            self.best_value = metric_value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience and self.patience > 0:
                self.stop = True
        return improved

    @property
    def best(self):
        return self.best_value

    @property
    def bad_streak(self):
        return self.num_bad