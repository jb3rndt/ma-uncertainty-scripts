

import re


NUMBER_REGEX = re.compile(r"^\-?\d+(\.\d+)?")

def is_number(value: str | int | float) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def extract_number(value: str | float | int):
    if isinstance(value, float) or isinstance(value, int):
        return float(value)
    match = NUMBER_REGEX.match(value)
    if match:
        return float(match.group(0))
    return None


def is_integer(value: str | float | int) -> bool:
    number = extract_number(value)
    return number.is_integer() if number is not None else False

def try_is_between(
    value: str | float | int, min_value: float, max_value: float
) -> bool:
    number = extract_number(value)
    if number is None:
        return False
    return min_value <= number <= max_value
