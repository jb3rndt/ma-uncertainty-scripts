from time import time

import pandas as pd

from src.downstream_task.config import RegressionConfig
from src.utils import get_raw_results


def prepare_data(config: RegressionConfig, dataset: str):
    df_raw = get_raw_results()

    cleaned_results = df_raw[
        (df_raw["dataset"] == dataset)
        & (df_raw["metric"] == "consistency_ruleBasedPipino")
        & (df_raw["dimension"] == "consistency_tuple")
        & (df_raw["type"] == "cleaned")
    ]
    polluted_results = df_raw[
        (df_raw["dataset"] == dataset)
        & (df_raw["metric"] == "consistency_ruleBasedPipino")
        & (df_raw["dimension"] == "consistency_tuple")
        & (df_raw["type"] == "polluted")
        & (df_raw["pollution_mechanism"] == "ECAR")
        & (df_raw["pollution_rate"] == 0.35)
    ]

    assert len(cleaned_results) == 1, cleaned_results
    assert len(polluted_results) == 1, polluted_results

    cleaned_data = pd.DataFrame(
        cleaned_results.iloc[0]["result"].data,
        columns=cleaned_results.iloc[0]["raw_column"].split(","),
    )[config.feature_cols + [config.target_col]]

    polluted_data = pd.DataFrame(
        polluted_results.iloc[0]["result"].data,
        columns=polluted_results.iloc[0]["raw_column"].split(","),
    )[config.feature_cols + [config.target_col]]

    polluted_dq = pd.Series(
        polluted_results.iloc[0]["result"].dq_result,
    )
    polluted_certainty = pd.Series(
        polluted_results.iloc[0]["result"].certainty,
    )

    return cleaned_data, polluted_data, polluted_dq, polluted_certainty


def eval_permutations(
    config: RegressionConfig,
    cleaned_data: pd.DataFrame,
    polluted_data: pd.DataFrame,
    polluted_dq: pd.Series,
    polluted_certainty: pd.Series,
):
    for n in range(config.n_runs):
        run_start_time = time()
        for data, key in [(cleaned_data, "cleaned"), (polluted_data, "polluted")]:
            yield (data, key, n, None)
        print(f"Completed run {n + 1} after {time() - run_start_time:.2f} seconds")

    for i, t in enumerate(config.thresholds):
        start_time = time()
        print(f"[{i+1}/{len(config.thresholds)}] Running for threshold {t}...")
        datasets = [
            (polluted_data.loc[polluted_dq >= t], "filtered_dq"),
            (
                polluted_data.loc[polluted_dq * polluted_certainty >= t],
                "filtered_dq_certainty",
            ),
        ]

        for n in range(config.n_runs):
            run_start_time = time()
            for data, key in datasets:
                if len(data) == 0:
                    continue
                yield (data, key, n, t)
            print(
                f"[{i+1}/{len(config.thresholds)}] Completed run {n + 1} after {time() - run_start_time:.2f} seconds"
            )
        print(
            f"[{i+1}/{len(config.thresholds)}] Completed after {time() - start_time:.2f} seconds"
        )
