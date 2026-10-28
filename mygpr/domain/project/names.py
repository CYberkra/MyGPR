"""Display names are metadata, never filesystem identities."""

import unicodedata


def display_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("名称必须是文字。")
    name = str(value).strip()
    if not name or len(name) > 80 or any(unicodedata.category(c) == "Cc" for c in name):
        raise ValueError("名称须为 1–80 个字符，不能包含换行或控制字符。")
    return name
