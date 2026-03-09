from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ColumnRawData:
    pollution_ratio: float
    pollution_mechanism: str | None
    data: List
    dq_result: List
    certainty: List
    is_clean: List


@dataclass
class ColumnEvaluationResult:
    pollution_ratio: float
    pollution_mechanism: str | None
    dq_results_null_ratio: float
    certainty_null_ratio: float
    mse: float
    mse_weighted: float
    pr_auc: float
    precision: List[float]
    recall: List[float]
    thresholds: List[float]
    pr_auc_weighted: float
    precision_weighted: List[float]
    recall_weighted: List[float]
    thresholds_weighted: List[float]
    fp: int
    fn: int
    tp: int
    tn: int
    fp_weighted: int
    fn_weighted: int
    tp_weighted: int
    tn_weighted: int
    js_divergence_per_method_and_model: Dict[str, Dict[str, float]]
    js_divergence_per_method_and_model_weighted: Dict[str, Dict[str, float]]
