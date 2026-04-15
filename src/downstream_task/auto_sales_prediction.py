import time
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.downstream_task.config import RegressionConfig
from src.downstream_task.utils import eval_permutations, prepare_data


def evaluate_classifier(
    config: RegressionConfig,
    data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    random_state: np.random.RandomState,
):
    train_idx, test_idx = train_test_split(
        data.index,
        test_size=config.test_size,
        random_state=random_state,
    )
    X_train = data.loc[train_idx].drop(config.target_col, axis=1)
    y_train = data.loc[train_idx][config.target_col]
    X_test = cleaned_data.loc[test_idx].drop(config.target_col, axis=1)
    y_test = cleaned_data.loc[test_idx][config.target_col]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = ensemble.GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        max_depth=config.max_depth,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_split=config.min_samples_split,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def evaluate_auto_sales_prediction(config: RegressionConfig):
    random_state = np.random.RandomState(config.random_seed)

    cleaned_data, polluted_data, polluted_dq, polluted_certainty = prepare_data(
        config, "auto_sales"
    )

    measurements: List[Dict[Literal["data", "score", "run", "threshold"], Any]] = []

    for data, key, n, t in eval_permutations(
        config, cleaned_data, polluted_data, polluted_dq, polluted_certainty
    ):
        measurements.append(
            {
                "data": key,
                "score": evaluate_classifier(config, data, cleaned_data, random_state),
                "run": n,
                "threshold": t,
            }
        )

    return measurements
