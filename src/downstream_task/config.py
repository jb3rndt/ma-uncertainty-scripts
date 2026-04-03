import dataclasses
from typing import List


@dataclasses.dataclass
class RegressionConfig:
    cols: List[str]
    n_runs: int = 10
    random_seed: int | None = None
    learning_rate: float = 0.2
    subsample: float = 0.5
    n_estimators: int = 400
    test_size: float = 0.2
    max_depth: int | None = None
    max_leaf_nodes: int | None = None
    min_samples_split: int = 2
    thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
