import pandas as pd


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
