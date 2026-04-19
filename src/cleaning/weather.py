import pandas as pd

from src.validation.dates import contains_expected_datetime_format
from src.validation.numbers import (
    extract_number,
    is_number,
    try_is_between,
)
from src.validation.strings import is_unpadded_nonempty_str

TEMP_RULES = [
    lambda value: try_is_between(value, -10, 50),
    lambda value: str(extract_number(value))[::-1].find(".") == 1,
    lambda value: is_number(value),
]
DIRECTIONS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


WEATHER_ORIGINAL_CONSISTENCY_RULES = {
    "Date": [
        is_unpadded_nonempty_str,
        lambda value: contains_expected_datetime_format(value.strip(), "%Y-%m-%d"),
    ],
    "Location": [
        is_unpadded_nonempty_str,
    ],
    "MinTemp": TEMP_RULES,
    "MaxTemp": TEMP_RULES,
    "Rainfall": [
        lambda value: isinstance(value, float),
        lambda value: value >= 0,
    ],
    "WindGustDir": [lambda value: value in DIRECTIONS],
    "WindGustSpeed": [
        lambda value: isinstance(value, float),
        lambda value: value.is_integer(),
        lambda value: value >= 0,
    ],
    "WindDir9am": [lambda value: value in DIRECTIONS],
    "WindDir3pm": [lambda value: value in DIRECTIONS],
    "WindSpeed9am": [
        lambda value: isinstance(value, float),
        lambda value: value.is_integer(),
        lambda value: value >= 0,
    ],
    "WindSpeed3pm": [
        lambda value: isinstance(value, float),
        lambda value: value.is_integer(),
        lambda value: value >= 0,
    ],
    "Humidity9am": [
        lambda value: isinstance(value, float),
        lambda value: value.is_integer(),
        lambda value: 0 <= value <= 100,
    ],
    "Humidity3pm": [
        lambda value: isinstance(value, float),
        lambda value: value.is_integer(),
        lambda value: 0 <= value <= 100,
    ],
    "Pressure9am": [
        lambda value: isinstance(value, float),
        lambda value: 950 <= value <= 1050,
        lambda value: value == round(value, 1),
    ],
    "Pressure3pm": [
        lambda value: isinstance(value, float),
        lambda value: 950 <= value <= 1050,
        lambda value: value == round(value, 1),
    ],
    "Temp9am": [*TEMP_RULES],
    "Temp3pm": [*TEMP_RULES],
    "RainToday": [
        lambda value: value in ["Yes", "No"],
    ],
    "RainTomorrow": [
        lambda value: value in ["Yes", "No"],
    ],
}

CONSISTENCY_TUPLE_RULES = [
    lambda row: row["MinTemp"] <= row["MaxTemp"],
    lambda row: row["MinTemp"] <= row["Temp9am"] <= row["MaxTemp"],
    lambda row: row["MinTemp"] <= row["Temp3pm"] <= row["MaxTemp"],
    lambda row: row["Rainfall"] != 0
    and row["RainToday"] == "Yes"
    or row["Rainfall"] == 0
    and row["RainToday"] == "No",
]


def clean_weather(data: pd.DataFrame) -> pd.DataFrame:
    # Drop null values
    cleaned = (
        data.replace("NA", pd.NA)
        .drop(columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"])
        .dropna()
    )

    cleaned["MinTemp"] = cleaned[["MinTemp", "Temp9am", "Temp3pm"]].min(axis=1)
    cleaned["MaxTemp"] = cleaned[["MaxTemp", "Temp9am", "Temp3pm"]].max(axis=1)

    cleaned["RainToday"] = cleaned["Rainfall"].apply(
        lambda x: "Yes" if x != 0 else "No"
    )

    for col, rules in WEATHER_ORIGINAL_CONSISTENCY_RULES.items():
        rule_results = cleaned[col].apply(
            lambda value: all(rule(value) for rule in rules)
        )
        if not rule_results.all():
            print(cleaned[~rule_results])
            raise ValueError(f"There are still consistency violations in column {col}.")

    # Apply tuple consistency rules
    for i, rule in enumerate(CONSISTENCY_TUPLE_RULES):
        rule_results = cleaned.apply(rule, axis="columns")
        if not rule_results.all():
            print(cleaned[~rule_results])
            raise ValueError(
                f"There are still consistency violations in the DataFrame. Rule #{i + 1} failed."
            )

    return cleaned
