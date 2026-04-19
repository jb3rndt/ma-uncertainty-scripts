from pathlib import Path
from typing import Any, Callable, List

import pandas as pd

from metis.metric.completeness.completeness_nullAndDMVRatio import (
    completeness_nullAndDMVRatio,
)
from metis.metric.completeness.completeness_nullAndDMVRatio_config import (
    completeness_nullAndDMVRatio_config,
    completeness_nullAndDMVRatio_config_dismis,
)
from metis.metric.completeness.completeness_nullRatio import completeness_nullRatio
from metis.metric.completeness.completeness_nullRatio_config import (
    completeness_nullRatio_config,
)
from metis.metric.config import DatasetDependentMetricConfig, MetricConfig
from metis.metric.consistency.consistency_ruleBasedPipino import (
    consistency_ruleBasedPipino,
)
from metis.metric.consistency.consistency_ruleBasedPipino_config import (
    consistency_ruleBasedPipino_config,
)
from metis.metric.correctness.correctness_heinrich import correctness_heinrich
from metis.metric.correctness.correctness_heinrich_config import (
    correctness_heinrich_config,
)
from metis.metric.timeliness.timeliness_heinrich import timeliness_heinrich
from metis.metric.timeliness.timeliness_heinrich_config import (
    timeliness_heinrich_column_config,
    timeliness_heinrich_config,
)
from metis.utils.datetime.datetime_precision import determine_datetime_precision
from src.cleaning.auto_sales import AUTO_SALES_ORIGINAL_CONSISTENCY_RULES
from src.cleaning.movies import MOVIES_ORIGINAL_CONSISTENCY_RULES, unpack_json_list
from src.cleaning.weather import WEATHER_ORIGINAL_CONSISTENCY_RULES
from src.constants import (
    ALLOWED_GENRES,
    TOP_OL_COLUMNS,
)
from src.utils import execute_run
from src.validation.dates import contains_expected_datetime_format, is_datetime
from src.validation.numbers import extract_number, is_integer, is_number, try_is_between


def notna(value: Any) -> bool:
    return pd.notna(value)


temp_rules = [
    lambda value: notna(value) and try_is_between(value, -10, 50),
    lambda value: notna(value) and str(extract_number(value))[::-1].find(".") == 1,
    lambda value: notna(value) and is_number(value),
]

speed_rules = [
    lambda value: notna(value) and try_is_between(value, 0, 140),
    lambda value: notna(value) and str(extract_number(value))[::-1].find(".") == 1,
    lambda value: notna(value) and is_number(value),
]

pressure_rules = [
    lambda value: notna(value) and try_is_between(value, 900, 1100),
    lambda value: notna(value) and str(extract_number(value))[::-1].find(".") == 1,
    lambda value: notna(value) and is_number(value),
]

humidity_rules = [
    # Changing [0-100] to [0-1] is undetectable here
    lambda value: notna(value) and try_is_between(value, 0, 100),
    lambda value: notna(value) and str(extract_number(value))[::-1].find(".") == 1,
    lambda value: notna(value) and is_number(value),
]


def assess_consistency(folder: Path, force=False):
    is_unpadded_str: Callable[[Any], bool] = (
        lambda value: notna(value) and value.strip() == value
    )
    no_semis = lambda value: notna(value) and ";" not in value

    metrics = [consistency_ruleBasedPipino.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        DatasetDependentMetricConfig(
            config_per_dataset={
                r"weather.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    column_rules={
                        "MinTemp": temp_rules,
                        "MaxTemp": temp_rules,
                        "Temp9am": temp_rules,
                        "Temp3pm": temp_rules,
                        "WindGustSpeed": speed_rules,
                        "WindSpeed9am": speed_rules,
                        "WindSpeed3pm": speed_rules,
                        "Pressure9am": pressure_rules,
                        "Pressure3pm": pressure_rules,
                        "Humidity9am": humidity_rules,
                        "Humidity3pm": humidity_rules,
                    },
                ),
                r"movies.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    column_rules={
                        "runtime": [
                            lambda value: notna(value)
                            and str(value).strip() == str(value),
                            lambda value: notna(value) and is_number(value),
                            lambda value: notna(value)
                            and try_is_between(value, 0, 300),
                        ],
                        "release_date": [
                            lambda value: notna(value)
                            and contains_expected_datetime_format(value, "%Y-%m-%d"),
                            lambda value: notna(value)
                            and determine_datetime_precision(value) == "day",
                        ],
                        "keywords": [is_unpadded_str, no_semis],
                        "production_companies": [is_unpadded_str, no_semis],
                        "genres": [
                            is_unpadded_str,
                            no_semis,
                            lambda value: notna(value)
                            and all(
                                genre in ALLOWED_GENRES for genre in value.split(",")
                            ),
                        ],
                    },
                ),
                r"auto_sales.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    column_rules={
                        "PRICEEACH": [
                            lambda value: notna(value) and is_number(value),
                            lambda value: notna(value) and not is_integer(value),
                        ],
                        "ORDERDATE": [
                            lambda value: notna(value) and value.strip() == value,
                            lambda value: (
                                notna(value)
                                and is_datetime(value.strip(), {"dayfirst": False})
                                and contains_expected_datetime_format(
                                    value.strip(), "%d/%m/%Y"
                                )
                            ),
                        ],
                    },
                ),
            }
        ),
    ]

    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        force=force,
    )


def assess_consistency_original_dataset(folder: Path, force=False):
    def unpack_json_each(key: str, rules: List):
        return [lambda value: rule(unpack_json_list(value, key)) for rule in rules]

    metrics = [consistency_ruleBasedPipino.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        DatasetDependentMetricConfig(
            config_per_dataset={
                r"weather.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    column_rules=WEATHER_ORIGINAL_CONSISTENCY_RULES,
                ),
                r"movies.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    column_rules={
                        **MOVIES_ORIGINAL_CONSISTENCY_RULES,
                        "genres": unpack_json_each(
                            "name", MOVIES_ORIGINAL_CONSISTENCY_RULES["genres"]
                        ),
                        "keywords": unpack_json_each(
                            "name", MOVIES_ORIGINAL_CONSISTENCY_RULES["keywords"]
                        ),
                        "production_countries": unpack_json_each(
                            "iso_3166_1",
                            MOVIES_ORIGINAL_CONSISTENCY_RULES["production_countries"],
                        ),
                        "spoken_languages": unpack_json_each(
                            "iso_639_1",
                            MOVIES_ORIGINAL_CONSISTENCY_RULES["spoken_languages"],
                        ),
                    },
                ),
                r"auto_sales.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    column_rules=AUTO_SALES_ORIGINAL_CONSISTENCY_RULES,
                ),
            }
        ),
    ]

    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        force=force,
    )


def assess_tuple_consistency_original_dataset(folder: Path, force=False):
    metrics = [consistency_ruleBasedPipino.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        DatasetDependentMetricConfig(
            config_per_dataset={
                r"weather.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    tuple_rules=[
                        lambda row: row["MinTemp"] <= row["MaxTemp"],
                        lambda row: row["MinTemp"] <= row["Temp9am"] <= row["MaxTemp"],
                        lambda row: row["MinTemp"] <= row["Temp3pm"] <= row["MaxTemp"],
                        lambda row: row["Rainfall"] != 0
                        and row["RainToday"] == "Yes"
                        or row["Rainfall"] == 0
                        and row["RainToday"] == "No",
                    ],
                ),
                r"movies.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    tuple_rules=[
                        lambda row: try_is_between(row["vote_average"], 0, 10),
                        lambda row: try_is_between(row["runtime"], 0, 300),
                    ],
                ),
                r"auto_sales.*": consistency_ruleBasedPipino_config(
                    skip_null_values=True,
                    tuple_rules=[
                        lambda row: row["PRICEEACH"] <= row["SALES"]
                        and row["MSRP"] <= row["SALES"],
                        lambda row: row["QUANTITYORDERED"] * row["PRICEEACH"]
                        == row["SALES"],
                    ],
                ),
            }
        ),
    ]

    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        force=force,
    )


def assess_tuple_consistency(folder: Path, force=False):
    metrics = [consistency_ruleBasedPipino.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        DatasetDependentMetricConfig(
            config_per_dataset={
                r"weather.*": consistency_ruleBasedPipino_config(
                    tuple_rules=[
                        lambda row: row["MinTemp"] <= row["MaxTemp"],
                        lambda row: try_is_between(row["Pressure9am"], 900, 1100)
                        and try_is_between(row["Pressure3pm"], 900, 1100)
                        and try_is_between(row["WindGustSpeed"], 0, 140)
                        and try_is_between(row["WindSpeed9am"], 0, 140)
                        and try_is_between(row["WindSpeed3pm"], 0, 140)
                        and try_is_between(row["Temp9am"], -10, 50)
                        and try_is_between(row["Temp3pm"], -10, 50)
                        and try_is_between(row["MinTemp"], -10, 50)
                        and try_is_between(row["MaxTemp"], -10, 50),
                        lambda row: row["Rainfall"] != 0
                        and row["RainToday"] == "Yes"
                        or row["Rainfall"] == 0
                        and row["RainToday"] == "No",
                    ],
                ),
                r"movies.*": consistency_ruleBasedPipino_config(
                    tuple_rules=[
                        lambda row: try_is_between(row["vote_average"], 0, 10),
                        lambda row: try_is_between(row["runtime"], 0, 300),
                        lambda row: round(row["popularity"], 6) == row["popularity"],
                    ],
                ),
                r"auto_sales.*": consistency_ruleBasedPipino_config(
                    tuple_rules=[
                        lambda row: row["PRICEEACH"] <= row["SALES"]
                        and row["MSRP"] <= row["SALES"],
                        lambda row: round(row["DAYS_SINCE_LASTORDER"])
                        == row["DAYS_SINCE_LASTORDER"],
                        lambda row: row["QUANTITYORDERED"] * row["PRICEEACH"]
                        == row["SALES"],
                    ],
                ),
            }
        ),
    ]

    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        force=force,
    )


def assess_completeness(folder: Path, force=False):
    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=[
            completeness_nullAndDMVRatio.__name__,
            completeness_nullRatio.__name__,
            completeness_nullAndDMVRatio.__name__,
        ],
        metric_configs=[
            DatasetDependentMetricConfig(
                config_per_dataset={
                    r"weather.*": completeness_nullAndDMVRatio_config(
                        dismis_config=completeness_nullAndDMVRatio_config_dismis(
                            value_embeddings_path=str(
                                folder / "weather.polluted_value_embeddings.json"
                            ),
                            example_dmvs_path="/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/weather_example_dmvs_detection.json",
                            example_embeddings_path="/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/weather_precomputed_example_embeddings.json",
                            column_types={
                                "Date": "date",
                                "Location": "categorical",
                                "MinTemp": "numeric",
                                "MaxTemp": "numeric",
                                "Rainfall": "numeric",
                                "WindGustDir": "categorical",
                                "WindGustSpeed": "numeric",
                                "WindDir9am": "categorical",
                                "WindDir3pm": "categorical",
                                "WindSpeed9am": "numeric",
                                "WindSpeed3pm": "numeric",
                                "Humidity9am": "numeric",
                                "Humidity3pm": "numeric",
                                "Pressure9am": "numeric",
                                "Pressure3pm": "numeric",
                                "Temp9am": "numeric",
                                "Temp3pm": "numeric",
                                "RainToday": "categorical",
                                "RainTomorrow": "categorical",
                            },
                        ),
                    ),
                    r"movies.*": completeness_nullAndDMVRatio_config(
                        dismis_config=completeness_nullAndDMVRatio_config_dismis(
                            value_embeddings_path=str(
                                folder / "movies.polluted_value_embeddings.json"
                            ),
                            example_dmvs_path="/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/movies_example_dmvs_detection.json",
                            example_embeddings_path="/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/movies_precomputed_example_embeddings.json",
                            column_types={
                                "id": "numeric",
                                "budget": "numeric",
                                "genres": "text",
                                "keywords": "text",
                                "original_language": "categorical",
                                "original_title": "categorical",
                                "overview": "text",
                                "popularity": "numeric",
                                "production_companies": "text",
                                "production_countries": "text",
                                "release_date": "date",
                                "revenue": "numeric",
                                "runtime": "numeric",
                                "spoken_languages": "text",
                                "status": "categorical",
                                "title": "text",
                                "vote_average": "numeric",
                                "vote_count": "numeric",
                            },
                        ),
                    ),
                    r"auto_sales.*": completeness_nullAndDMVRatio_config(
                        dismis_config=completeness_nullAndDMVRatio_config_dismis(
                            value_embeddings_path=str(
                                folder / "auto_sales.polluted_value_embeddings.json"
                            ),
                            example_dmvs_path="/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/auto_sales_example_dmvs_detection.json",
                            example_embeddings_path="/Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/auto_sales_precomputed_example_embeddings.json",
                            column_types={
                                "ORDERNUMBER": "numeric",
                                "QUANTITYORDERED": "numeric",
                                "PRICEEACH": "numeric",
                                "ORDERLINENUMBER": "numeric",
                                "SALES": "numeric",
                                "ORDERDATE": "date",
                                "DAYS_SINCE_LASTORDER": "numeric",
                                "STATUS": "categorical",
                                "PRODUCTLINE": "categorical",
                                "MSRP": "numeric",
                                "PRODUCTCODE": "categorical",
                                "CUSTOMERNAME": "text",
                                "PHONE": "text",
                                "ADDRESSLINE1": "text",
                                "CITY": "text",
                                "POSTALCODE": "text",
                                "COUNTRY": "text",
                                "CONTACTLASTNAME": "text",
                                "CONTACTFIRSTNAME": "text",
                                "DEALSIZE": "categorical",
                            },
                        ),
                    ),
                }
            ),
            completeness_nullAndDMVRatio_config(),
            completeness_nullRatio_config(),
        ],
        force=force,
    )


def assess_correctness(folder: Path, force=False):
    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=[correctness_heinrich.__name__],
        metric_configs=[
            DatasetDependentMetricConfig(
                config_per_dataset={
                    r"weather.*": correctness_heinrich_config(
                        reference_file_path=folder / "weather.reference.csv",
                        superset_file_path=folder / "weather.superset.csv",
                    ),
                    r"movies.*": correctness_heinrich_config(
                        reference_file_path=folder / "movies.reference.csv",
                        superset_file_path=folder / "movies.superset.csv",
                    ),
                    r"auto_sales.*": correctness_heinrich_config(
                        reference_file_path=folder / "auto_sales.reference.csv",
                        superset_file_path=folder / "auto_sales.superset.csv",
                    ),
                }
            ),
        ],
        force=force,
    )


def assess_timeliness(folder: Path, force=False):
    weather_relevance_interval = 365.25
    metrics = [timeliness_heinrich.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        DatasetDependentMetricConfig(
            config_per_dataset={
                r"weather.*": timeliness_heinrich_config(
                    timeliness_config_per_column={
                        col: timeliness_heinrich_column_config(
                            decline_rate=decline_rate,
                            ingestion_date_column="Date",
                            to_datetime_kwargs={"format": "mixed"},
                            simulated_assessment_date="2017-07-01",  # newest entry in weather data: 2017-06-25
                        )
                        for col, decline_rate in [
                            ("MinTemp", 1 / weather_relevance_interval),
                            ("MaxTemp", 1 / weather_relevance_interval),
                            ("Rainfall", 1 / weather_relevance_interval),
                            ("Temp9am", (24 / 6) / weather_relevance_interval),
                            ("Temp3pm", (24 / 18) / weather_relevance_interval),
                            ("WindGustSpeed", 1 / weather_relevance_interval),
                            ("WindSpeed9am", (24 / 6) / weather_relevance_interval),
                            ("WindSpeed3pm", (24 / 18) / weather_relevance_interval),
                            ("Pressure9am", (24 / 6) / weather_relevance_interval),
                            ("Pressure3pm", (24 / 18) / weather_relevance_interval),
                            ("Humidity9am", (24 / 6) / weather_relevance_interval),
                            ("Humidity3pm", (24 / 18) / weather_relevance_interval),
                        ]
                    }
                ),
                r"open_library.*": timeliness_heinrich_config(
                    timeliness_config_per_column={
                        col: timeliness_heinrich_column_config(
                            decline_rate=stats["avg_changes"] / stats["avg_time"],
                            ingestion_date_column="last_modified",
                            to_datetime_kwargs={"format": "ISO8601"},
                            simulated_assessment_date="2026-01-01",  # newest entry in open library data: 2025-12-31T22:00:00.274823
                            # simulated_timestamp_precision="year",
                        )
                        for col, stats in TOP_OL_COLUMNS
                    }
                ),
                r"auto_sales.*": timeliness_heinrich_config(
                    timeliness_config_per_column={
                        col: timeliness_heinrich_column_config(
                            decline_rate=0.1 / 365.25,
                            ingestion_date_column="ORDERDATE",
                            to_datetime_kwargs={
                                "dayfirst": True,
                                "format": "mixed",
                            },
                            simulated_assessment_date="2021-05-30",  # newest entry in auto sales data: 2020-05-30T22:00:00.000Z
                            # simulated_timestamp_precision="year",
                        )
                        for col in ["ADDRESSLINE1", "CITY", "POSTALCODE", "COUNTRY"]
                    }
                ),
            }
        ),
    ]

    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        force=force,
    )
