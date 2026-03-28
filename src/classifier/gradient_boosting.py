import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import ensemble
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
    min_samples_split: int = 2
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


def evaluate_classifier(
    config: RegressionConfig,
    random_state: np.random.RandomState | None,
    data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
):
    train_idx, test_idx = train_test_split(
        data.index,
        test_size=config.test_size,
        stratify=cleaned_data.loc[data.index]["RainTomorrow"],
        random_state=random_state,
    )
    X_train = data.loc[train_idx].drop("RainTomorrow", axis=1)
    y_train = data.loc[train_idx]["RainTomorrow"]
    X_test = cleaned_data.loc[test_idx].drop("RainTomorrow", axis=1)
    y_test = cleaned_data.loc[test_idx]["RainTomorrow"]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = ensemble.GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        max_depth=config.max_depth,
        random_state=config.random_state,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_split=config.min_samples_split,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def run():
    run = int(time.time())
    Path(f"regression-results/{run}").mkdir(exist_ok=True, parents=True)
    config = RegressionConfig(
        random_state=run,
        max_leaf_nodes=4,
        min_samples_split=5,
        test_size=0.3,
        n_estimators=300,
        max_depth=None,
        subsample=0.5,
        learning_rate=0.2,
    )
    random_state = np.random.RandomState(config.random_state)

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
        & (df_raw["pollution_rate"] == 0.35)
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

    measurements: List[Dict[Literal["data", "score", "run", "threshold"], Any]] = []

    for n in range(config.n_runs):
        run_start_time = time.time()
        measurements.append(
            {
                "data": "cleaned",
                "score": evaluate_classifier(
                    config, random_state, cleaned_data, cleaned_data
                ),
                "run": n,
                "threshold": None,
            }
        )
        measurements.append(
            {
                "data": "polluted",
                "score": evaluate_classifier(
                    config, random_state, polluted_data, cleaned_data
                ),
                "run": n,
                "threshold": None,
            }
        )
        print(f"Completed run {n + 1} after {time.time() - run_start_time:.2f} seconds")

    for i, t in enumerate(config.thresholds):
        start_time = time.time()
        print(f"[{i+1}/{len(config.thresholds)}] Running for threshold {t}...")
        datasets = [
            (polluted_data.loc[polluted_dq > t], "filtered_dq"),
            (
                polluted_data.loc[polluted_dq * polluted_certainty > t],
                "filtered_dq_certainty",
            ),
        ]

        for n in range(config.n_runs):
            run_start_time = time.time()
            for data, key in datasets:
                if len(data) == 0:
                    continue
                measurements.append(
                    {
                        "data": key,
                        "score": evaluate_classifier(
                            config, random_state, data, cleaned_data
                        ),
                        "run": n,
                        "threshold": t,
                    }
                )
            print(
                f"[{i+1}/{len(config.thresholds)}] Completed run {n + 1} after {time.time() - run_start_time:.2f} seconds"
            )
        print(
            f"[{i+1}/{len(config.thresholds)}] Completed after {time.time() - start_time:.2f} seconds"
        )

    df_gb = pd.DataFrame(measurements)
    df_gb.to_csv(f"regression-results/{run}/results.csv", index=False)
    json.dump(
        dataclasses.asdict(config),
        open(f"regression-results/{run}/config.json", "w"),
        indent=2,
    )
