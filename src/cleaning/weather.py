import pandas as pd

from src.assessment import (
    extract_number,
    is_number,
    is_unpadded_nonempty_str,
    try_is_between,
)

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


CONSISTENCY_RULES = {
    "row ID": [
        is_unpadded_nonempty_str,
        lambda value: value.startswith("Row") and is_number(value[3:]),
    ],
    "Location": [
        is_unpadded_nonempty_str,
    ],
    "MinTemp": [*TEMP_RULES],
    "MaxTemp": [*TEMP_RULES],
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


def clean_weather(data: pd.DataFrame) -> pd.DataFrame:
    # Drop null values
    cleaned = data.drop(
        columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
    ).dropna()

    # RainTomorrow stores 0 and 1 instead of Yes and No
    cleaned["RainTomorrow"] = cleaned["RainTomorrow"].map({0: "No", 1: "Yes"})

    for col, rules in CONSISTENCY_RULES.items():
        rule_results = cleaned[col].apply(
            lambda value: all(rule(value) for rule in rules)
        )
        if not rule_results.all():
            print(cleaned[~rule_results])
            raise ValueError(f"There are still consistency violations in column {col}.")

    return cleaned
