from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from src.downstream_task.config import RegressionConfig
from src.downstream_task.utils import generate_eval_permutations, prepare_data


def encode_data(cleaned_data: pd.DataFrame, polluted_data: pd.DataFrame):
    one_hot_cols = ["original_language", "status"]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(polluted_data[one_hot_cols])
    encoded_cleaned_data = pd.DataFrame(
        np.array(enc.transform(cleaned_data[one_hot_cols])),
        columns=enc.get_feature_names_out(cleaned_data[one_hot_cols].columns),
        index=cleaned_data.index,
    ).join(cleaned_data.drop(one_hot_cols, axis=1))
    encoded_polluted_data = pd.DataFrame(
        np.array(enc.transform(polluted_data[one_hot_cols])),
        columns=enc.get_feature_names_out(polluted_data[one_hot_cols].columns),
        index=polluted_data.index,
    ).join(polluted_data.drop(one_hot_cols, axis=1))

    multi_label_cols = ["genres", "production_countries"]

    for col in multi_label_cols:
        encoded_cleaned_data[col] = encoded_cleaned_data[col].apply(
            lambda x: x.split(",")
        )
        encoded_polluted_data[col] = encoded_polluted_data[col].apply(
            lambda x: x.split(",")
        )
        mlb = MultiLabelBinarizer()
        mlb.fit(encoded_polluted_data[col])
        encoded_cleaned_data = pd.DataFrame(
            np.array(mlb.transform(encoded_cleaned_data[col])),
            columns=mlb.classes_,
            index=encoded_cleaned_data.index,
        ).join(encoded_cleaned_data.drop(col, axis=1))
        encoded_polluted_data = pd.DataFrame(
            np.array(mlb.transform(encoded_polluted_data[col])),
            columns=mlb.classes_,
            index=encoded_polluted_data.index,
        ).join(encoded_polluted_data.drop(col, axis=1))

    return encoded_cleaned_data, encoded_polluted_data


def evaluate_classifier(
    config: RegressionConfig,
    data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
):
    index = data.sample(frac=1, random_state=config.random_state).index
    train_idx, test_idx = train_test_split(
        index,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    X_train = data.loc[train_idx].drop(config.target_col, axis=1)
    y_train = data.loc[train_idx][config.target_col]
    X_test = cleaned_data.loc[test_idx].drop(config.target_col, axis=1)
    y_test = cleaned_data.loc[test_idx][config.target_col]

    # scaler = StandardScaler()
    # scaler.fit(X_train)

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    clf = ensemble.GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        max_depth=config.max_depth,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # y_pred = np.full_like(y_test, y_train.mean())
    return np.sqrt(mean_squared_error(y_test, y_pred))


def evaluate_movies_prediction(config: RegressionConfig):
    (
        cleaned_data,
        polluted_data,
        polluted_dq,
        polluted_certainty,
    ) = prepare_data(config, "movies")

    cleaned_data, polluted_data = encode_data(cleaned_data, polluted_data)

    measurements: List[
        Dict[Literal["data", "score", "run", "threshold", "dataset_size"], Any]
    ] = []

    for data, key, n, t in generate_eval_permutations(
        config,
        cleaned_data,
        polluted_data,
        polluted_dq,
        polluted_certainty,
        dataset_size=1858,
    ):
        measurements.append(
            {
                "data": key,
                "score": evaluate_classifier(config, data, cleaned_data),
                "run": n,
                "threshold": t,
                "dataset_size": len(data),
            }
        )

    return measurements
