import json
import re
from datetime import datetime

import pandas as pd
import pycountry
import requests

from metis.utils.datetime.datetime_precision import determine_datetime_precision
from src.constants import (
    ALLOWED_COUNTRIES,
    ALLOWED_GENRES,
    ALLOWED_LANGUAGES,
)
from src.validation.dates import contains_expected_datetime_format
from src.validation.numbers import is_integer, is_number
from src.validation.strings import is_unpadded_nonempty_str

MOVIES_ORIGINAL_CONSISTENCY_RULES = {
    "budget": [is_number, lambda value: int(value) > 0, is_integer],
    "genres": [
        is_unpadded_nonempty_str,
        lambda value: all(genre in ALLOWED_GENRES for genre in value.split(",")),
    ],
    "id": [is_number, is_integer, lambda value: value > 0],
    "keywords": [is_unpadded_nonempty_str],
    "original_language": [
        is_unpadded_nonempty_str,
        lambda value: value in ALLOWED_LANGUAGES,
    ],
    "original_title": [is_unpadded_nonempty_str],
    "overview": [is_unpadded_nonempty_str],
    "popularity": [is_number, lambda value: float(value) >= 0],
    "production_companies": [is_unpadded_nonempty_str],
    "production_countries": [
        is_unpadded_nonempty_str,
        lambda value: all(lang in ALLOWED_COUNTRIES for lang in value.split(",")),
    ],
    "release_date": [
        is_unpadded_nonempty_str,
        lambda value: contains_expected_datetime_format(value, "%Y-%m-%d"),
        lambda value: determine_datetime_precision(value) == "day",
    ],
    "revenue": [is_number, lambda value: int(value) > 0, is_integer],
    "runtime": [is_number, lambda value: int(value) > 0, is_integer],
    "spoken_languages": [
        is_unpadded_nonempty_str,
        lambda value: all(lang in ALLOWED_LANGUAGES for lang in value.split(",")),
    ],
    "status": [
        is_unpadded_nonempty_str,
        lambda value: value
        in [
            "Rumored",
            "Post Production",
            "Released",
        ],
    ],
    "title": [is_unpadded_nonempty_str],
    "vote_average": [is_number, lambda value: 0 <= float(value) <= 10],
    "vote_count": [is_number, lambda value: int(value) >= 0, is_integer],
}


def unpack_json_list(value: str, key: str):
    if pd.isna(value) or value.strip() == "":
        return value
    json_data = json.loads(value)
    return ",".join([d[key].strip() for d in json_data])


def clean_movies(original: pd.DataFrame) -> pd.DataFrame:
    # Drop null values
    cleaned = original.drop(columns=["homepage", "tagline"]).dropna()

    # Extract json data
    json_cols = [
        ("genres", "name"),
        ("keywords", "name"),
        ("production_companies", "name"),
        ("spoken_languages", "iso_639_1"),
        ("production_countries", "iso_3166_1"),
    ]
    for col, key in json_cols:
        cleaned[col] = cleaned[col].apply(lambda x: unpack_json_list(x, key))

    # Drop consistency violations
    cleaned = cleaned[cleaned["genres"] != ""]
    cleaned = cleaned[cleaned["keywords"] != ""]
    cleaned = cleaned[cleaned["production_companies"] != ""]
    cleaned = cleaned[cleaned["production_countries"] != ""]
    cleaned = cleaned[cleaned["spoken_languages"] != ""]
    cleaned = cleaned[cleaned["spoken_languages"] != "xx"]
    cleaned = cleaned[cleaned["revenue"] > 0]
    cleaned = cleaned[cleaned["budget"] > 0]
    cleaned = cleaned[cleaned["runtime"] > 0]
    cleaned["runtime"] = cleaned["runtime"].astype(int)

    # print(
    #     set(
    #         cleaned["spoken_languages"].str.split(",").explode().unique().tolist()
    #         + cleaned["original_language"].str.split(",").explode().unique().tolist()
    #     )
    # )

    if (cleaned.notna().mean() < 1).any():
        raise ValueError("There are still null values in the cleaned dataset.")
    for col, rules in MOVIES_ORIGINAL_CONSISTENCY_RULES.items():
        rule_results = cleaned[col].apply(
            lambda value: all(rule(value) for rule in rules)
        )
        if not rule_results.all():
            print(cleaned[~rule_results][["id", col]])
            raise ValueError(f"There are still consistency violations in column {col}.")

    print("Size reduced by", 100 - len(cleaned) / len(original) * 100, "%")
    return cleaned
