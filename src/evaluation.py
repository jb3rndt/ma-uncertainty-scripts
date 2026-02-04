import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd

from src.utils import grouped_results_and_certainties


def mse_per_column(expected: pd.DataFrame, predicted: pd.DataFrame) -> pd.Series:
    errors = (expected - predicted) ** 2
    return errors.mean()


def evaluate_run(results_folder: Path):
    data_configs: List[Tuple[Path, str]] = [
        (file, file.name.replace(".loader_config.json", ""))
        for file in results_folder.rglob("*.loader_config.json")
    ]

    reference_per_data_config: Dict[
        str,
        Dict[Literal["data", "mask"], pd.DataFrame],
    ] = {}

    for path, dataset in data_configs:
        with open(path, "r") as f:
            data_config = json.load(f)
        data = pd.read_csv(data_config["file_name"])
        mask_file = data_config["file_name"].replace(".csv", ".mask.csv")
        if not Path(mask_file).exists():
            print(f"No mask found for {dataset}")
            mask = pd.DataFrame(False, data.index, data.columns)
        else:
            mask = pd.read_csv(mask_file)
        reference_per_data_config[dataset] = {
            "data": data,
            "mask": mask,
        }

    evaluations = {}

    dq_results_flat = pd.read_csv(results_folder / "dq_results.csv")

    for dataset, metric, dq_results, dq_certainties in grouped_results_and_certainties(
        dq_results_flat
    ):
        data = reference_per_data_config[dataset]["data"]
        if any("," in col for col in dq_results.columns.to_list()):
            print("Evaluating tuple base aggregation")
            mask = ~reference_per_data_config[dataset]["mask"]
            mask = pd.DataFrame(mask.mean(axis=1), columns=dq_results.columns)
        else:
            # mask has to be inverted because True indicates polluted values for which the quality should be 0
            mask = ~reference_per_data_config[dataset]["mask"][
                dq_results.columns.tolist()
            ]
            data = data[dq_results.columns.tolist()]

        evaluations.setdefault(metric, {}).setdefault(dataset, {})
        evaluations[metric][dataset]["dataset_stats"] = {
            "describe": data.describe().to_dict(),
            "null_rates": data.isna().mean().to_dict(),
        }
        evaluations[metric][dataset]["pollution_rates"] = (1 - mask.mean()).to_dict()
        evaluations[metric][dataset]["dq_results_stats"] = {
            "describe": dq_results.describe().to_dict(),
            "null_rates": dq_results.isna().mean().to_dict(),
        }
        evaluations[metric][dataset]["dq_certainties_stats"] = {
            "describe": dq_certainties.describe().to_dict(),
            "null_rates": dq_certainties.isna().mean().to_dict(),
        }

        mse_per_column_values_no_quality = mse_per_column(
            mask,
            pd.DataFrame(1.0, index=dq_results.index, columns=dq_results.columns),
        )
        mse_per_column_values = mse_per_column(mask, dq_results)
        mse_per_column_values_with_certainty = mse_per_column(
            mask, dq_results * dq_certainties
        )
        evaluations[metric][dataset]["mse_per_column"] = pd.DataFrame(
            {
                "Without quality values": mse_per_column_values_no_quality,
                "With quality values": mse_per_column_values,
                "With certainty weighting": mse_per_column_values_with_certainty,
            }
        ).to_dict()

        # print(evaluate_aggregation_methods(dq_results, dq_certainties, mask))

        # plot_accuracy_by_threshold(dq_results, ~mask, dq_results.columns.tolist())
        # plot_f1_score_by_threshold(dq_results, ~mask, dq_results.columns.tolist())
        # plot_pr_auc_curve(dq_results, ~mask, dq_results.columns.tolist())

    evaluations_file = results_folder / "evaluations.json"
    json.dump(evaluations, open(evaluations_file, "w"), indent=2)
    print(f"Saved evaluations to {evaluations_file.absolute()}")
    return evaluations
