import json
import re
from datetime import datetime

import pandas as pd
import pycountry
import requests

from metis.utils.datetime.datetime_precision import determine_datetime_precision
from src.assessment import (
    contains_expected_datetime_format,
    get_datetime_part,
    is_datetime_with_location,
    is_duration_format,
    is_min_abbr,
    is_minute_unit,
    is_number,
    location_is_at_end,
)

PERSON_NAME_REGEX = r"[\w\-\.\&']+(\s[\w\-\.\&']+)*"
LANGUAGE_NAME_REGEX = r"[\w\-\(\)]+(\s[\w\-\(\)]+)*"
PERSON_LIST_REGEX = rf"^{PERSON_NAME_REGEX}(,{PERSON_NAME_REGEX})*$"
# fmt: off
ALLOWED_GENRES = {"Action", "History", "Fantasy", "Adult", "Biography", "Comedy", "Musical", "Romance", "Sport", "Drama", "News", "Family", "Sci-Fi", "Western", "War", "Documentary", "Film-Noir", "Mystery", "Adventure", "Music", "Thriller", "Short", "Crime", "Horror", "Animation"}
# fmt: on

CONSISTENCY_RULES = {
    "Release Date": [
        lambda value: is_datetime_with_location(value),
        lambda value: location_is_at_end(value),
        lambda value: contains_expected_datetime_format(
            get_datetime_part(value), "%d %B %Y"  # e.g., "25 December 2020"
        ),
        lambda value: determine_datetime_precision(get_datetime_part(value)) == "day",
    ],
    "Duration": [
        lambda value: is_duration_format(value),
        lambda value: is_minute_unit(value),
        lambda value: is_min_abbr(value),
    ],
    "Year": [is_number],
    "Id": [lambda value: value.startswith("tt") and is_number(value[2:])],
    "Name": [lambda value: isinstance(value, str) and len(value) > 0],
    "Director": [lambda value: isinstance(value, str) and len(value) > 0],
    "Creator": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: re.match(PERSON_LIST_REGEX, value) is not None,
    ],
    "Actors": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: re.match(PERSON_LIST_REGEX, value) is not None,
    ],
    "Cast": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: re.match(PERSON_LIST_REGEX, value) is not None,
    ],
    "Language": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: re.match(
            rf"^{LANGUAGE_NAME_REGEX}(,{LANGUAGE_NAME_REGEX})*$", value
        )
        is not None,
    ],
    "Country": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: re.match(r"^[A-Za-z\s]+(,[A-Za-z\s]+)*$", value) is not None,
    ],
    "RatingValue": [is_number],
    "RatingCount": [is_number],
    "ReviewCount": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: re.match(
            r"^((\d[\d,]*\s(user,\d[\d,]*\scritic|user))|(\d[\d,]*\scritic)?)$", value
        )
        is not None,
    ],  # e.g. 414 user,177 critic
    "Genre": [
        lambda value: isinstance(value, str) and len(value) > 0,
        lambda value: all(genre in ALLOWED_GENRES for genre in value.split(",")),
    ],
    "Description": [lambda value: isinstance(value, str) and len(value) > 0],
}

LANGUAGE_PATCHES = {
    "tt0096928": "English,French,German,Greek,Ancient (to 1453)",
    "tt0814255": "English,Greek,Ancient (to 1453)",
    "tt0462465": "English,Norse,Old,Latin",
}


def clean_movies(original: pd.DataFrame) -> pd.DataFrame:
    # Drop null values
    cleaned = original.drop(columns=["Filming Locations"]).dropna()

    # Solve consistency violations
    with open("patches_selected.json", "r") as f:
        patches = json.load(f)
    violated_tuples = cleaned[
        cleaned["Release Date"].apply(
            lambda value: not all(
                rule(value) for rule in CONSISTENCY_RULES["Release Date"]
            )
        )
    ]
    for _, row in violated_tuples.iterrows():
        patch = patches[row["Id"]]
        if patch.get("drop"):
            cleaned = cleaned[cleaned["Id"] != row["Id"]]
        else:
            cleaned.loc[cleaned["Id"] == row["Id"], "Release Date"] = patch["selected"]
            cleaned.loc[cleaned["Id"] == row["Id"], "Year"] = int(
                patch["selected"].split(" (")[0].split(" ")[-1]
            )

    for id, language in LANGUAGE_PATCHES.items():
        cleaned.loc[cleaned["Id"] == id, "Language"] = language

    if (cleaned.notna().mean() < 1).any():
        raise ValueError("There are still null values in the cleaned dataset.")
    for col, rules in CONSISTENCY_RULES.items():
        rule_results = cleaned[col].apply(
            lambda value: all(rule(value) for rule in rules)
        )
        if not rule_results.all():
            print(cleaned[~rule_results][["Id", col]])
            raise ValueError(f"There are still consistency violations in column {col}.")

    return cleaned


def list_release_date_patches(original: pd.DataFrame):
    data = original.drop(columns=["Filming Locations"]).dropna()
    patches = json.load(open("patches.json"))

    violated_tuples = data[
        data["Release Date"].apply(
            lambda value: not all(
                rule(value) for rule in CONSISTENCY_RULES["Release Date"]
            )
        )
    ]
    for _, row in violated_tuples.iterrows():
        if row["Id"] in patches:
            continue  # Skip already patched entries
        release_dates = fetch_release_dates(row["Id"])
        patches[row["Id"]] = {
            "original": row["Release Date"],
            "fetched": [
                {"date": date["releaseDate"], "country": date.get("country", {})}
                for date in release_dates
            ],
        }

    print(
        f"Found {len(violated_tuples)} consistency violations in column 'Release Date'."
    )
    with open("patches.json", "w") as f:
        json.dump(patches, f, indent=2)


def format_release_date_json(date):
    month = date["date"].get("month")
    return f"{date['date'].get('day', '')} {datetime(1, month, 1).strftime('%B') if month else ''} {date['date'].get('year', '')} ({date.get("country", {}).get("code")})".strip()


def convert_country_code_to_name(code):
    if code == "XWG":
        return "West Germany"
    country = pycountry.countries.get(alpha_2=code)
    if not country:
        raise ValueError(f"Unknown country code: {code}")
    return country.name


def select_closest_release_date():
    with open("patches.json", "r") as f:
        patches = json.load(f)

    for _, patch in patches.items():
        dates = [
            (
                datetime(
                    date["date"].get("year"),
                    date["date"].get("month"),
                    date["date"].get("day"),
                ),
                date.get("country", {}).get("code"),
            )
            for date in patch["fetched"]
            if all(key in date["date"] for key in ["year", "month", "day"])
            and date.get("country", {}).get("code") is not None
        ]
        formatted_dates = [format_release_date_json(date) for date in patch["fetched"]]
        patch["fetched"] = formatted_dates
        if dates:
            selected = min(
                dates,
                key=lambda date: date[0],
            )
            patch["selected"] = (
                selected[0].strftime("%d %B %Y").lstrip("0")
                + " ("
                + convert_country_code_to_name(selected[1])
                + ")"
            )
        else:
            patch["drop"] = True

    with open("patches_selected.json", "w") as f:
        json.dump(patches, f, indent=2)


def fetch_release_dates(id: str):
    url = f"https://api.imdbapi.dev/titles/{id}/releaseDates"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("releaseDates", [])
    else:
        print(
            f"Failed to fetch release dates for ID {id}. Status code: {response.status_code}"
        )
        return []
