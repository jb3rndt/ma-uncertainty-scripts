import dataclasses
import json
import warnings
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Tuple

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from src.evaluation.aggregation import evaluate_aggregation_methods
from src.evaluation.types import ColumnEvaluationResult, ColumnRawData
from src.utils import grouped_results_and_certainties


def is_polluted_dataset(dataset_name: str) -> bool:
    return not any(keyword in dataset_name for keyword in ["original", "cleaned"])


def get_error_mechanism(mechanism_config_value: str):
    possible = ["EAR", "ENAR", "ECAR"]
    for mechanism in possible:
        if mechanism in mechanism_config_value:
            return mechanism
    raise ValueError(
        f"Unknown mechanism in config value {mechanism_config_value}. Expected one of {possible}"
    )


def mse(expected: pd.Series, predicted: pd.Series):
    errors = (expected - predicted) ** 2
    return errors.mean()


def pr_auc_per_column(expected: pd.DataFrame, predicted: pd.DataFrame) -> Dict:
    return {
        column: pr_auc(expected[column], predicted[column])
        for column in expected.columns
    }


PR_AUC_RESULT = NamedTuple(
    "Result",
    [("precision", List), ("recall", List), ("thresholds", List), ("pr_auc", float)],
)


def pr_auc(expected: pd.Series, predicted: pd.Series):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="sklearn.metrics._ranking"
        )
        precision, recall, thresholds = precision_recall_curve(expected, predicted)
        pr_auc = auc(recall, precision)
        return PR_AUC_RESULT(
            precision.tolist(),
            recall.tolist(),
            thresholds.tolist(),
            float(pr_auc),
        )


def evaluate_run(
    results_folder: Path,
    skip_existing_evaluations: bool = False,
):
    if (
        skip_existing_evaluations
        and (results_folder / "evaluations.json").exists()
        and (results_folder / "raw_results.json").exists()
    ):
        print(
            f"Evaluations and raw results already exist for {results_folder}. Skipping evaluation."
        )
        return

    data_configs: List[Tuple[Path, str]] = [
        (file, file.name.replace(".loader_config.json", ""))
        for file in results_folder.rglob("*.loader_config.json")
    ]

    reference_per_data_config: Dict[
        str,
        Dict[Literal["data", "mask"], pd.DataFrame],
    ] = {}
    pollution_configs: Dict[str, Dict] = {}

    for path, dataset in data_configs:
        with open(path, "r") as f:
            data_config = json.load(f)
        data = pd.read_csv(data_config["file_name"])
        mask_file = data_config["file_name"].replace(".csv", ".mask.csv")
        pollution_config = data_config["file_name"].replace(".csv", ".config.json")
        if is_polluted_dataset(dataset):
            is_polluted_mask = pd.read_csv(mask_file)
            pollution_configs[dataset] = json.load(open(pollution_config, "r"))
        else:
            is_polluted_mask = pd.DataFrame(False, data.index, data.columns)
        reference_per_data_config[dataset] = {
            "data": data,
            "mask": is_polluted_mask,
        }

    raw_results = {}
    evaluations = {}

    dq_results_flat = pd.read_csv(results_folder / "dq_results.csv")

    for dataset, metric, dq_results, dq_certainties in grouped_results_and_certainties(
        dq_results_flat
    ):
        data = reference_per_data_config[dataset]["data"]
        pollution_config = pollution_configs.get(dataset, None)
        if any("," in col for col in dq_results.columns.to_list()):
            is_clean_mask = ~reference_per_data_config[dataset]["mask"]
            is_clean_mask = pd.DataFrame(
                is_clean_mask.mean(axis=1), columns=dq_results.columns
            )
        else:
            # mask has to be inverted because True indicates polluted values for which the quality should be 0
            is_clean_mask = ~reference_per_data_config[dataset]["mask"][
                dq_results.columns.tolist()
            ]
            data = data[dq_results.columns.tolist()]

        raw_results.setdefault(metric, {}).setdefault(dataset, {})
        raw_results[metric][dataset] = {
            col: dataclasses.asdict(
                ColumnRawData(
                    pollution_ratio=1 - is_clean_mask[col].mean(),
                    pollution_mechanism=next(
                        iter(
                            {
                                get_error_mechanism(config["error_mechanism"])
                                for config in (pollution_config or {})
                                .get("columns", {})
                                .get(col, [])
                            }
                        ),
                        None,
                    ),
                    data=data[col].to_list(),
                    dq_result=dq_results[col].to_list(),
                    certainty=dq_certainties[col].to_list(),
                    is_clean=is_clean_mask[col].to_list(),
                )
            )
            for col in dq_results.columns
        }

        # TP, FP, TN, FN rates per column
        FP = ((is_clean_mask == 1) & (dq_results < 1)).sum()
        FN = ((is_clean_mask == 0) & (dq_results == 1)).sum()
        TP = ((is_clean_mask == 0) & (dq_results < 1)).sum()
        TN = ((is_clean_mask == 1) & (dq_results == 1)).sum()
        FPW = ((is_clean_mask == 1) & (dq_results * dq_certainties < 1)).sum()
        FNW = ((is_clean_mask == 0) & (dq_results * dq_certainties == 1)).sum()
        TPW = ((is_clean_mask == 0) & (dq_results * dq_certainties < 1)).sum()
        TNW = ((is_clean_mask == 1) & (dq_results * dq_certainties == 1)).sum()

        evaluations.setdefault(metric, {}).setdefault(dataset, {})

        for col in dq_results.columns:
            # sklearn requires binary labels for precision-recall curve
            binary_is_polluted_mask = is_clean_mask[col] < 1
            pr_auc_result = pr_auc(binary_is_polluted_mask, 1 - dq_results[col])
            pr_auc_result_weighted = pr_auc(
                binary_is_polluted_mask, 1 - dq_results[col] * dq_certainties[col]
            )

            aggregation_results = evaluate_aggregation_methods(
                dq_results[col], dq_certainties[col], 1 - is_clean_mask[col]
            )

            evaluations[metric][dataset][col] = dataclasses.asdict(
                ColumnEvaluationResult(
                    pollution_ratio=1 - is_clean_mask[col].mean(),
                    pollution_mechanism=next(
                        iter(
                            {
                                get_error_mechanism(config["error_mechanism"])
                                for config in (pollution_config or {})
                                .get("columns", {})
                                .get(col, [])
                            }
                        ),
                        None,
                    ),
                    dq_results_null_ratio=dq_results[col].isna().mean(),
                    certainty_null_ratio=dq_certainties[col].isna().mean(),
                    mse=mse(is_clean_mask[col], dq_results[col]),
                    mse_weighted=mse(
                        is_clean_mask[col], dq_results[col] * dq_certainties[col]
                    ),
                    pr_auc=pr_auc_result.pr_auc,
                    precision=pr_auc_result.precision,
                    recall=pr_auc_result.recall,
                    thresholds=pr_auc_result.thresholds,
                    pr_auc_weighted=pr_auc_result_weighted.pr_auc,
                    precision_weighted=pr_auc_result_weighted.precision,
                    recall_weighted=pr_auc_result_weighted.recall,
                    thresholds_weighted=pr_auc_result_weighted.thresholds,
                    fp=int(FP[col]),
                    fn=int(FN[col]),
                    tp=int(TP[col]),
                    tn=int(TN[col]),
                    fp_weighted=int(FPW[col]),
                    fn_weighted=int(FNW[col]),
                    tp_weighted=int(TPW[col]),
                    tn_weighted=int(TNW[col]),
                    js_divergence_per_method_and_model=aggregation_results[
                        "divergence"
                    ],
                    js_divergence_per_method_and_model_weighted=aggregation_results[
                        "weighted_divergence"
                    ],
                )
            )

        # print(evaluate_aggregation_methods(dq_results, dq_certainties, mask))

        # plot_accuracy_by_threshold(dq_results, ~mask, dq_results.columns.tolist())
        # plot_f1_score_by_threshold(dq_results, ~mask, dq_results.columns.tolist())
        # plot_pr_auc_curve(dq_results, ~mask, dq_results.columns.tolist())

    evaluations_file = results_folder / "evaluations.json"
    json.dump(evaluations, open(evaluations_file, "w"), indent=2)

    raw_results_file = results_folder / "raw_results.json"
    json.dump(raw_results, open(raw_results_file, "w"))

    print(
        f"Saved raw results to {raw_results_file.absolute()} and evaluations to {evaluations_file.absolute()}"
    )
