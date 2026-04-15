from typing import Any


def is_unpadded_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == value and len(value) > 0
