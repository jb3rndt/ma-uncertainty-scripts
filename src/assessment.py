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
from metis.metric.timeliness.timeliness_heinrich import timeliness_heinrich
from metis.metric.timeliness.timeliness_heinrich_config import (
    timeliness_heinrich_config,
)
from metis.utils.datetime.datetime_precision import determine_datetime_precision
from src.constants import ALLOWED_GENRES, OL_COLUMN_CHANGE_RATES, PERSON_LIST_REGEX
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


def is_unpadded_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == value and len(value) > 0


def is_datetime(value: str) -> bool:
    try:
        pd.to_datetime(value)
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
    if (folder / "results" / "dq_results.csv").exists() and not force:
        print(
            f"Results file {(folder / 'results' / 'dq_results.csv').absolute()} already exists. Skipping."
        )
        return folder / "results"

    comma_delimited_checks: List[Callable[[Any], bool]] = [
        lambda value: notna(value) and value.strip() == value,
        lambda value: notna(value)
        and re.match(PERSON_LIST_REGEX, value.strip()) is not None,
        lambda value: notna(value)
        and any((word or "x")[0].isupper() for word in value.strip().split(" ")),
    ]

    metrics = [consistency_ruleBasedPipino.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        consistency_ruleBasedPipino_config(
            attribute_rules={
                "MinTemp": temp_rules,
                "MaxTemp": temp_rules,
                "PRICEEACH": [
                    lambda value: notna(value) and is_number(value),
                    lambda value: notna(value)
                    and is_number(value)
                    or (not is_number(value) and "$" in value),
                ],
                "ORDERDATE": [
                    lambda value: notna(value) and is_datetime(value),
                    lambda value: (
                        notna(value)
                        and is_datetime(value)
                        and contains_expected_datetime_format(value, "%d/%m/%Y")
                    ),
                    lambda value: (
                        notna(value)
                        and is_datetime(value)
                        and determine_datetime_precision(value) == "day"
                    ),
                ],
                "Id": [
                    lambda value: str(value).startswith("tt"),
                    lambda value: len(str(value)) == 9,
                    lambda value: is_number(str(value)[-7:]),
                ],
                "Actors": comma_delimited_checks,
                "Cast": comma_delimited_checks,
                "Duration": [
                    lambda value: notna(value) and value.strip() == value,
                    lambda value: notna(value)
                    and is_duration_format(value.strip())
                    and is_minute_unit(value.strip()),
                    lambda value: notna(value)
                    and is_duration_format(value.strip())
                    and is_min_abbr(value.strip()),
                ],
                "Release Date": [
                    lambda value: notna(value) and is_datetime_with_location(value),
                    lambda value: notna(value) and location_is_at_end(value),
                    lambda value: notna(value)
                    and contains_expected_datetime_format(
                        get_datetime_part(value), "%d %B %Y"  # e.g., "25 December 2020"
                    ),
                    lambda value: notna(value)
                    and determine_datetime_precision(get_datetime_part(value)) == "day",
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
    )


def assess_completeness(folder: Path, force=False):
    if (folder / "results").exists() and not force:
        print(
            f"Results folder {(folder / 'results').absolute()} already exists. SKIPPING!"
        )
        return folder / "results"

    return execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=[
            completeness_nullAndDMVRatio.__name__,
            completeness_nullRatio.__name__,
        ],
        metric_configs=[
            completeness_nullAndDMVRatio_config(
                aggregation_axis="columns", aggregate_all=False
            ),
            completeness_nullRatio_config(
                aggregation_axis="columns", aggregate_all=False
            ),
        ],
    )


def assess_timeliness(folder: Path, force=False):
    if (folder / "results").exists() and not force:
        print(
            f"Results folder {(folder / 'results').absolute()} already exists. SKIPPING!"
        )
        return folder / "results"

    metrics = [timeliness_heinrich.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        timeliness_heinrich_config(
            decline_rate_per_column={
                "ADDRESSLINE1": 0.1 / 365.25,
            },
            ingestion_date_column="ORDERDATE",
            to_datetime_kwargs={"dayfirst": True},
            simulated_assessment_date="2021-05-30",  # newest entry in auto sales data: 2020-05-30T22:00:00.000Z
        ),
    ]

    results_folder_one = execute_run(
        results_folder=folder / "results_AS",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        datasets=["auto_sales"],
    )

    metrics = [timeliness_heinrich.__name__]
    metric_configs: List[str | None | MetricConfig] = [
        timeliness_heinrich_config(
            decline_rate_per_column={
                col: stats["avg_changes"] / stats["avg_time"]
                for col, stats in sorted(
                    list(OL_COLUMN_CHANGE_RATES.items()),
                    key=lambda item: (
                        item[1]["avg_changes"] / item[1]["avg_time"]
                        if item[1]["avg_time"] is not None
                        and item[1]["avg_changes"] is not None
                        and item[0]
                        not in [
                            "last_modified",
                            "latest_revision",
                            "revision",
                            "created",
                        ]
                        else 0
                    ),
                    reverse=True,
                )[:5]
                if stats["avg_time"] is not None and stats["avg_changes"] is not None
            },
            ingestion_date_column="last_modified",
            to_datetime_kwargs={"format": "ISO8601"},
            simulated_assessment_date="2026-01-01",  # newest entry in open library data: 2025-12-31T22:00:00.274823
            simulated_timestamp_precision="year",
        ),
    ]

    results_folder_two = execute_run(
        results_folder=folder / "results",
        polluted_folder=folder,
        metrics=metrics,
        metric_configs=metric_configs,
        datasets=["open_library"],
    )

    if not results_folder_one or not results_folder_two:
        print("One of the assessments did not produce results. Skipping combination.")
        return results_folder_one or results_folder_two

    df_one = pd.read_csv(results_folder_one / "dq_results.csv")
    df_two = pd.read_csv(results_folder_two / "dq_results.csv")
    combined_df = pd.concat([df_one, df_two], ignore_index=True)
    # Save everything into second folder and remove first folder
    combined_df.to_csv(results_folder_two / "dq_results.csv", index=False)
    for file in results_folder_one.glob("*"):
        prefix = "one_" if (results_folder_two / file.name).exists() else ""
        if file.name != "dq_results.csv":
            file.rename(results_folder_two / f"{prefix}{file.name}")
    (results_folder_one / "dq_results.csv").unlink()
    results_folder_one.rmdir()
    return results_folder_two
