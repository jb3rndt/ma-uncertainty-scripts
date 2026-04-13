import re
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd

from metis.metric.completeness.completeness_nullAndDMVRatio import (
    completeness_nullAndDMVRatio,
)
from metis.metric.completeness.completeness_nullAndDMVRatio_config import (
    completeness_nullAndDMVRatio_config,
)
from metis.metric.completeness.completeness_nullRatio import completeness_nullRatio
from metis.metric.completeness.completeness_nullRatio_config import (
    completeness_nullRatio_config,
)
from metis.metric.config import MetricConfig
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
from src.constants import (
    ALLOWED_GENRES,
    CLEANED_DATA_PATH,
    PERSON_LIST_REGEX,
    TOP_OL_COLUMNS,
)
from src.utils import execute_run

NUMBER_REGEX = re.compile(r"^\-?\d+(\.\d+)?")


def extract_number(value: str | float | int):
    if isinstance(value, float) or isinstance(value, int):
        return float(value)
    match = NUMBER_REGEX.match(value)
    if match:
        return float(match.group(0))
    return None


def notna(value: Any) -> bool:
    return pd.notna(value)


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_integer(value: str | float | int) -> bool:
    number = extract_number(value)
    if number is None:
        return False
    return number.is_integer()


def is_unpadded_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == value and len(value) > 0


def is_datetime(value: str, to_datetime_kwargs={}) -> bool:
    try:
        pd.to_datetime(value, **to_datetime_kwargs)
        return True
    except (ValueError, TypeError):
        return False


def contains_expected_datetime_format(value: str, format: str) -> bool:
    try:
        pd.to_datetime(value, exact=False, format=format)
        return True
    except (ValueError, TypeError):
        return False


def try_is_between(
    value: str | float | int, min_value: float, max_value: float
) -> bool:
    number = extract_number(value)
    if number is None:
        return False
    return min_value <= number <= max_value


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


def is_duration_format(value: str) -> bool:
    parts = value.split(" ")
    if len(parts) != 2:
        return False
    number_part = parts[0]
    if not is_number(number_part):
        return False
    return True


def is_minute_unit(value: str) -> bool:
    parts = value.split(" ")
    if len(parts) != 2:
        return False
    unit_part = parts[1]
    return unit_part in ["min", "m"]


def is_min_abbr(value: str) -> bool:
    parts = value.split(" ")
    if len(parts) != 2:
        return False
    unit_part = parts[1]
    return unit_part == "min"


def get_datetime_part(value: str) -> str:
    location = re.search(r"\(.*\)", value)
    if not location:
        return value
    return value.replace(location.group(0), "").strip()


def is_datetime_with_location(value: str) -> bool:
    location = re.search(r"\(.*\)", value)
    if not location:
        return False
    dt_part = get_datetime_part(value)
    return is_datetime(dt_part)


def location_is_at_end(value: str) -> bool:
    match = re.search(r"\(.*\)", value)
    if not match:
        return False
    return match.end() == len(value)


def assess_consistency(folder: Path, force=False):
    comma_delimited_checks: List[Callable[[Any], bool]] = [
        lambda value: notna(value) and value.strip() == value,
        lambda value: notna(value)
        and re.match(PERSON_LIST_REGEX, value.strip()) is not None,
        # Detect false positives
        lambda value: notna(value)
        and all((word or "x")[0].isupper() for word in value.strip().split(" ")),
    ]

    metrics = [consistency_ruleBasedPipino.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        consistency_ruleBasedPipino_config(
            attribute_rules={
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
                "PRICEEACH": [
                    lambda value: notna(value) and is_number(value),
                    lambda value: notna(value) and not is_integer(value),
                ],
                "ORDERDATE": [
                    lambda value: notna(value) and value.strip() == value,
                    lambda value: (
                        notna(value)
                        and is_datetime(value.strip(), {"dayfirst": False})
                        and contains_expected_datetime_format(value.strip(), "%d/%m/%Y")
                    ),
                ],
                "Id": [
                    lambda value: value.startswith("tt"),
                    # lambda value: len(str(value)) == 9, trigger false negatives
                    lambda value: is_number(value.replace("tt", "")),
                ],
                "Actors": comma_delimited_checks,
                "Cast": comma_delimited_checks,
                "Duration": [
                    lambda value: notna(value) and value.strip() == value,
                    lambda value: notna(value)
                    and is_duration_format(value.strip())
                    and is_minute_unit(value.strip()),
                    # Trigger false negatives
                    # lambda value: notna(value)
                    # and is_duration_format(value.strip())
                    # and is_min_abbr(value.strip()),
                ],
                "Release Date": [
                    lambda value: notna(value) and is_datetime_with_location(value),
                    lambda value: notna(value) and location_is_at_end(value),
                    lambda value: notna(value)
                    and contains_expected_datetime_format(
                        get_datetime_part(value), "%d %B %Y"  # e.g., "25 December 2020"
                    ),
                ],
                "Genre": [
                    *comma_delimited_checks,
                    lambda value: notna(value)
                    and all(genre in ALLOWED_GENRES for genre in value.split(",")),
                ],
            },
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
        consistency_ruleBasedPipino_config(
            tuple_rules=[
                (["MinTemp", "MaxTemp"], lambda row: row["MinTemp"] <= row["MaxTemp"]),
                # (["MinTemp", "Temp9am", "MaxTemp"], lambda row: row["MinTemp"] <= row["Temp9am"] <= row["MaxTemp"]),
                # (["MinTemp", "Temp3pm", "MaxTemp"], lambda row: row["MinTemp"] <= row["Temp3pm"] <= row["MaxTemp"]),
                # Correct unit bounds
                (
                    [
                        "Pressure9am",
                        "Pressure3pm",
                        "WindGustSpeed",
                        "WindSpeed9am",
                        "WindSpeed3pm",
                        "Temp9am",
                        "Temp3pm",
                        "MinTemp",
                        "MaxTemp",
                    ],
                    lambda row: try_is_between(row["Pressure9am"], 900, 1100)
                    and try_is_between(row["Pressure3pm"], 900, 1100)
                    and try_is_between(row["WindGustSpeed"], 0, 140)
                    and try_is_between(row["WindSpeed9am"], 0, 140)
                    and try_is_between(row["WindSpeed3pm"], 0, 140)
                    and try_is_between(row["Temp9am"], -10, 50)
                    and try_is_between(row["Temp3pm"], -10, 50)
                    and try_is_between(row["MinTemp"], -10, 50)
                    and try_is_between(row["MaxTemp"], -10, 50),
                ),
                (
                    ["Rainfall", "RainToday"],
                    lambda row: row["Rainfall"] != 0
                    and row["RainToday"] == "Yes"
                    or row["Rainfall"] == 0
                    and row["RainToday"] == "No",
                ),
                (
                    ["PRICEEACH", "SALES"],
                    lambda row: try_is_between(row["PRICEEACH"], 20, 300)
                    and try_is_between(row["SALES"], 400, 14000),
                ),
                (
                    ["DAYS_SINCE_LASTORDER"],
                    (
                        lambda row: round(row["DAYS_SINCE_LASTORDER"])
                        == row["DAYS_SINCE_LASTORDER"]
                    ),
                ),
                (
                    ["QUANTITYORDERED", "PRICEEACH", "SALES"],
                    lambda row: row["QUANTITYORDERED"] * row["PRICEEACH"]
                    == row["SALES"],
                ),
                (["RatingValue", "RatingCount"], lambda row: try_is_between(row["RatingValue"], 0, 10)),
            ],
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
        ],
        metric_configs=[
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
        # Hard coded for weather data for now
        metric_configs=[
            correctness_heinrich_config(
                reference_file_path=folder / "weather.reference.csv",
                superset_file_path=CLEANED_DATA_PATH / "weather.csv",
            )
        ],
        force=force,
    )


def assess_timeliness(folder: Path, force=False):
    metrics = [timeliness_heinrich.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        timeliness_heinrich_config(
            timeliness_config_per_column={
                **{
                    col: timeliness_heinrich_column_config(
                        decline_rate=0.1 / 365.25,
                        ingestion_date_column="ORDERDATE",
                        to_datetime_kwargs={"dayfirst": True, "format": "mixed"},
                        simulated_assessment_date="2021-05-30",  # newest entry in auto sales data: 2020-05-30T22:00:00.000Z
                        # simulated_timestamp_precision="year",
                    )
                    for col in ["ADDRESSLINE1", "CITY", "POSTALCODE", "COUNTRY"]
                },
                **{
                    col: timeliness_heinrich_column_config(
                        decline_rate=stats["avg_changes"] / stats["avg_time"],
                        ingestion_date_column="last_modified",
                        to_datetime_kwargs={"format": "ISO8601"},
                        simulated_assessment_date="2026-01-01",  # newest entry in open library data: 2025-12-31T22:00:00.274823
                        # simulated_timestamp_precision="year",
                    )
                    for col, stats in TOP_OL_COLUMNS
                },
                **{
                    col: timeliness_heinrich_column_config(
                        decline_rate=decline_rate,
                        ingestion_date_column="Date",
                        to_datetime_kwargs={"format": "mixed"},
                        simulated_assessment_date="2017-07-01",  # newest entry in weather data: 2017-06-25
                    )
                    for col, decline_rate in [
                        ("MinTemp", 1 / 30),
                        ("MaxTemp", 1 / 30),
                        ("Rainfall", 1 / 30),
                        ("Temp9am", (24 / 6) / 30),
                        ("Temp3pm", (24 / 18) / 30),
                        ("WindGustSpeed", 1 / 30),
                        ("WindSpeed9am", (24 / 6) / 30),
                        ("WindSpeed3pm", (24 / 18) / 30),
                        ("Pressure9am", (24 / 6) / 30),
                        ("Pressure3pm", (24 / 18) / 30),
                        ("Humidity9am", (24 / 6) / 30),
                        ("Humidity3pm", (24 / 18) / 30),
                    ]
                },
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
