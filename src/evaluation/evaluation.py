import json
import warnings
from operator import is_
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from src.evaluation.aggregation import evaluate_aggregation_methods
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


def mse_per_column(expected: pd.DataFrame, predicted: pd.DataFrame) -> pd.Series:
    errors = (expected - predicted) ** 2
    return errors.mean()


def pr_auc_per_column(expected: pd.DataFrame, predicted: pd.DataFrame) -> Dict:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="sklearn.metrics._ranking"
        )
        pr_auc_values = {}
        for column in expected.columns:
            precision, recall, thresholds = precision_recall_curve(
                expected[column], predicted[column]
            )
            pr_auc = auc(recall, precision)
            pr_auc_values[column] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
                "pr_auc": pr_auc,
            }
        return pr_auc_values


def evaluate_run(
    results_folder: Path,
    callback: (
        Callable[
            [str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None
        ]
        | None
    ) = None,
):
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

        if callback:
            callback(dataset, metric, data, is_clean_mask, dq_results, dq_certainties)
            continue

        evaluations.setdefault(metric, {}).setdefault(dataset, {})
        evaluations[metric][dataset]["dataset_stats"] = {
            "describe": data.describe().to_dict(),
            "null_rates": data.isna().mean().to_dict(),
        }
        evaluations[metric][dataset]["pollution_rates"] = (
            1 - is_clean_mask.mean()
        ).to_dict()
        evaluations[metric][dataset]["pollution_mechanism"] = (
            {
                col: list(
                    {
                        get_error_mechanism(config["error_mechanism"])
                        for config in configs
                    }
                )
                for col, configs in pollution_config["columns"].items()
            }
            if pollution_config
            else {}
        )
        evaluations[metric][dataset]["dq_results_stats"] = {
            "describe": dq_results.describe().to_dict(),
            "null_rates": dq_results.isna().mean().to_dict(),
        }
        evaluations[metric][dataset]["dq_certainties_stats"] = {
            "describe": dq_certainties.describe().to_dict(),
            "null_rates": dq_certainties.isna().mean().to_dict(),
        }

        mse_per_column_values_no_quality = mse_per_column(
            is_clean_mask,
            pd.DataFrame(1.0, index=dq_results.index, columns=dq_results.columns),
        )
        mse_per_column_values = mse_per_column(is_clean_mask, dq_results)
        mse_per_column_values_with_certainty = mse_per_column(
            is_clean_mask, dq_results * dq_certainties
        )
        evaluations[metric][dataset]["mse_per_column"] = pd.DataFrame(
            {
                "Without quality values": mse_per_column_values_no_quality,
                "With quality values": mse_per_column_values,
                "With certainty weighting": mse_per_column_values_with_certainty,
            }
        ).to_dict()

        # sklearn requires binary labels for precision-recall curve
        binary_is_polluted_mask = is_clean_mask < 1
        evaluations[metric][dataset]["pr_auc_per_column"] = pr_auc_per_column(
            binary_is_polluted_mask, 1 - dq_results
        )
        evaluations[metric][dataset]["pr_auc_per_column_weighted"] = pr_auc_per_column(
            binary_is_polluted_mask, 1 - dq_results * dq_certainties
        )

        # TP, FP, TN, FN rates per column
        FP = ((is_clean_mask == 1) & (dq_results < 1)).sum()
        FN = ((is_clean_mask == 0) & (dq_results == 1)).sum()
        TP = ((is_clean_mask == 0) & (dq_results < 1)).sum()
        TN = ((is_clean_mask == 1) & (dq_results == 1)).sum()
        FPW = ((is_clean_mask == 1) & (dq_results * dq_certainties < 1)).sum()
        FNW = ((is_clean_mask == 0) & (dq_results * dq_certainties == 1)).sum()
        TPW = ((is_clean_mask == 0) & (dq_results * dq_certainties < 1)).sum()
        TNW = ((is_clean_mask == 1) & (dq_results * dq_certainties == 1)).sum()
        evaluations[metric][dataset]["false_positive_rate_per_column"] = (
            FP / (FP + TN)
        ).to_dict()
        evaluations[metric][dataset]["false_negative_rate_per_column"] = (
            FN / (FN + TP)
        ).to_dict()
        evaluations[metric][dataset]["false_positive_rate_per_column_weighted"] = (
            FPW / (FPW + TNW)
        ).to_dict()
        evaluations[metric][dataset]["false_negative_rate_per_column_weighted"] = (
            FNW / (FNW + TPW)
        ).to_dict()

        evaluations[metric][dataset]["aggregation_evaluation"] = (
            evaluate_aggregation_methods(dq_results, dq_certainties, 1 - is_clean_mask)
        )

        # print(evaluate_aggregation_methods(dq_results, dq_certainties, mask))

        # plot_accuracy_by_threshold(dq_results, ~mask, dq_results.columns.tolist())
        # plot_f1_score_by_threshold(dq_results, ~mask, dq_results.columns.tolist())
        # plot_pr_auc_curve(dq_results, ~mask, dq_results.columns.tolist())

    if not callback:
        evaluations_file = results_folder / "evaluations.json"
        json.dump(evaluations, open(evaluations_file, "w"), indent=2)
        print(f"Saved evaluations to {evaluations_file.absolute()}")
        return evaluations
