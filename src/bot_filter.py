import pandas as pd

TEMP_DOMAINS = {
    "@tempmail.com",
    "@10minutemail.com",
    "@mailinator.com",
    "@guerrillamail.com",
    "@discard.email",
}


def is_temp_domain(email: str) -> bool:
    if not isinstance(email, str):
        return False
    email = email.lower()
    return any(email.endswith(dom) for dom in TEMP_DOMAINS)


def is_bot(
    open_count: int | float = 0, click_count: int | float = 0, email: str = ""
) -> bool:
    open_count = 0 if open_count is None or pd.isna(open_count) else open_count
    click_count = 0 if click_count is None or pd.isna(click_count) else click_count

    if (open_count > 50 and click_count > 50) or is_temp_domain(email):
        return True
    return False
