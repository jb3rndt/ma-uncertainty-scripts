import dataclasses
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_raw_results


@dataclasses.dataclass
class RegressionConfig:
    n_runs: int = 10
    random_state: int | None = None
    learning_rate: float = 0.2
    subsample: float = 0.5
    n_estimators: int = 400
    test_size: float = 0.2
    max_depth: int | None = None
    max_leaf_nodes: int | None = None
    min_samples_split: int | None = None
    cols: List[str] = dataclasses.field(
        default_factory=lambda: [
            # "Humidity3pm",
            # "Humidity9am",
            "MaxTemp",
            "MinTemp",
            "Temp9am",
            "Temp3pm",
            "Pressure3pm",
            "Pressure9am",
            "RainToday",
            "RainTomorrow",
            "WindGustSpeed",
            "WindSpeed3pm",
            "WindSpeed9am",
            # "Pressure",
            # "Humidity",
            # "Temp",
            # "WindSpeed",
        ]
    )
    thresholds: List[float] = dataclasses.field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


def run():
    run = int(time.time())
    Path(f"regression-results/{run}").mkdir(exist_ok=True, parents=True)
    config = RegressionConfig(random_state=run, max_leaf_nodes=4, min_samples_split=5)

    df_raw = get_raw_results()

    cleaned_results = df_raw[
        (df_raw["dataset"] == "weather")
        & (df_raw["metric"] == "completeness_nullAndDMVRatio")
        & (df_raw["type"] == "cleaned")
    ]
    polluted_results = df_raw[
        (df_raw["dataset"] == "weather")
        & (df_raw["metric"] == "consistency_ruleBasedPipino")
        & (df_raw["dimension"] == "consistency_tuple")
        & (df_raw["type"] == "polluted")
        & (df_raw["pollution_mechanism"] == "ECAR")
        & (df_raw["pollution_rate"] == 0.2)
    ]

    cleaned_data = pd.DataFrame(
        {res["column"]: res["result"].data for _, res in cleaned_results.iterrows()}
    )

    assert len(polluted_results) == 1, polluted_results

    polluted_data = pd.DataFrame(
        polluted_results.iloc[0]["result"].data,
        columns="row ID,Location,MinTemp,MaxTemp,Rainfall,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Temp9am,Temp3pm,RainToday,RainTomorrow".split(
            ","
        ),
    )

    polluted_dq = pd.Series(
        polluted_results.iloc[0]["result"].dq_result,
    )
    polluted_certainty = pd.Series(
        polluted_results.iloc[0]["result"].certainty,
    )

    # Transformation
    def transform(data):
        data["RainToday"] = data["RainToday"].map({"Yes": 1, "No": 0})
        data["RainTomorrow"] = data["RainTomorrow"].map({"Yes": 1, "No": 0})
        # ["MaxTemp","MinTemp","Pressure3pm","Pressure9am","RainTomorrow","WindGustSpeed"]
        return data[config.cols]

    cleaned_data = transform(cleaned_data)
    polluted_data = transform(polluted_data)

    all_idx = cleaned_data.index
    results = {}

    for t in config.thresholds:
        datasets = [
            (cleaned_data, "clean"),
            (polluted_data, "polluted"),
            (polluted_data.loc[polluted_dq > t], "filtered_dq"),
            (
                polluted_data.loc[polluted_dq * polluted_certainty > t],
                "filtered_dq_certainty",
            ),
        ]

        run_scores = {key: [] for _, key in datasets}

        for _ in range(config.n_runs):
            train_idx, test_idx = train_test_split(
                all_idx,
                test_size=config.test_size,
                stratify=cleaned_data["RainTomorrow"],
            )

            test_df: pd.DataFrame = cleaned_data.loc[test_idx]
            X_test_clean = test_df.drop("RainTomorrow", axis=1)
            y_test_clean = test_df["RainTomorrow"]

            for data, key in datasets:
                train_idx_available = data.index.intersection(train_idx)
                if len(train_idx_available) == 0:
                    run_scores[key].append(np.nan)
                    continue

                train_df = data.loc[train_idx_available]
                X_train = train_df.drop("RainTomorrow", axis=1)
                y_train = train_df["RainTomorrow"]

                if y_train.nunique() < 2:
                    run_scores[key].append(np.nan)
                    continue

                scaler = StandardScaler()
                scaler.fit(X_train)

                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test_clean)

                clf = GradientBoostingClassifier(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    subsample=config.subsample,
                    max_depth=config.max_depth,
                ).fit(X_train_scaled, y_train)

                y_pred = clf.predict(X_test_scaled)
                run_scores[key].append(f1_score(y_test_clean, y_pred))

        for key, scores in run_scores.items():
            valid_scores = [s for s in scores if not np.isnan(s)]
            results.setdefault(key, []).append(
                np.mean(valid_scores) if valid_scores else np.nan
            )

    df_gb = pd.DataFrame(results, index=config.thresholds)
    df_gb.to_csv(f"regression-results/{run}/results.csv")
    json.dump(
        dataclasses.asdict(config),
        open(f"regression-results/{run}/config.json", "w"),
        indent=2,
    )
    df_gb.plot(kind="line")
    plt.show()
