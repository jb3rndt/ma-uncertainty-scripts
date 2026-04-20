import re

import pandas as pd

from metis.utils.datetime.datetime_precision import determine_datetime_precision
from src.validation.dates import contains_expected_datetime_format, is_datetime
from src.validation.strings import is_unpadded_nonempty_str

AUTO_SALES_ORIGINAL_CONSISTENCY_RULES = {
    "ORDERNUMBER": [
        lambda value: isinstance(value, int),
        lambda value: value >= 0,
        lambda value: len(str(int(value))) == 5,
    ],
    "QUANTITYORDERED": [
        lambda value: isinstance(value, int),
        lambda value: value > 0,
    ],
    "PRICEEACH": [
        lambda value: isinstance(value, float),
        lambda value: value > 0,
        lambda value: value == round(value, 2),
    ],
    "ORDERLINENUMBER": [
        lambda value: isinstance(value, int),
        lambda value: 1 <= value <= 20,
    ],
    "SALES": [
        lambda value: isinstance(value, float),
        lambda value: value > 0,
        lambda value: value == round(value, 2),
    ],
    "ORDERDATE": [
        lambda value: is_datetime(value),
        lambda value: (
            is_datetime(value) and contains_expected_datetime_format(value, "%d/%m/%Y")
        ),
        lambda value: (
            is_datetime(value) and determine_datetime_precision(value) == "day"
        ),
    ],
    "DAYS_SINCE_LASTORDER": [
        lambda value: isinstance(value, int),
        lambda value: value > 0,
    ],
    "STATUS": [
        lambda value: value
        in ["Shipped", "Disputed", "In Process", "Cancelled", "Resolved", "On Hold"]
    ],
    "PRODUCTLINE": [
        lambda value: value
        in [
            "Classic Cars",
            "Motorcycles",
            "Planes",
            "Ships",
            "Trains",
            "Trucks and Buses",
            "Vintage Cars",
        ]
    ],
    "MSRP": [lambda value: isinstance(value, int), lambda value: value > 0],
    "PRODUCTCODE": [lambda value: re.match(r"^S\d+_\d+$", value) is not None],
    "CUSTOMERNAME": [is_unpadded_nonempty_str],
    "PHONE": [lambda value: re.match(r"\d+", value) is not None],
    "ADDRESSLINE1": [is_unpadded_nonempty_str],
    "CITY": [is_unpadded_nonempty_str],
    "POSTALCODE": [is_unpadded_nonempty_str],
    "COUNTRY": [is_unpadded_nonempty_str],
    "CONTACTLASTNAME": [is_unpadded_nonempty_str],
    "CONTACTFIRSTNAME": [is_unpadded_nonempty_str],
    "DEALSIZE": [lambda value: value in ["Small", "Medium", "Large"]],
}

AUTO_SALES_ORIGINAL_CONSISTENCY_TUPLE_RULES = [
    lambda row: row["PRICEEACH"] <= row["SALES"] and row["MSRP"] <= row["SALES"],
    lambda row: round(row["DAYS_SINCE_LASTORDER"]) == row["DAYS_SINCE_LASTORDER"],
    lambda row: round(float(row["QUANTITYORDERED"]) * float(row["PRICEEACH"]), 2)
    == float(row["SALES"]),
]


def clean_auto_sales(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data

    cleaned["PHONE"] = cleaned["PHONE"].replace(r"\D", "", regex=True)
    cleaned["SALES"] = (cleaned["QUANTITYORDERED"] * cleaned["PRICEEACH"]).round(2)

    for col, rules in AUTO_SALES_ORIGINAL_CONSISTENCY_RULES.items():
        rule_results = cleaned[col].apply(
            lambda value: all(rule(value) for rule in rules)
        )
        if not rule_results.all():
            print(cleaned[~rule_results])
            raise ValueError(f"There are still consistency violations in column {col}.")

    for i, rule in enumerate(AUTO_SALES_ORIGINAL_CONSISTENCY_TUPLE_RULES):
        rule_results = cleaned.apply(lambda row: rule(row), axis=1)
        if not rule_results.all():
            print(cleaned[~rule_results])
            raise ValueError(
                f"There are still consistency violations in the dataset. (Rule {i})"
            )
    return cleaned
