"""Validation helpers for form input."""

from datetime import datetime


def validate_date_range(start_date: str, end_date: str, max_days: int = 3650) -> None:
    """Validate a pair of ISO date strings.

    Parameters
    ----------
    start_date, end_date : str
        Date strings in ``YYYY-MM-DD`` format.
    max_days : int, optional
        Maximum allowed range length in days. Defaults to ``3650``.

    Raises
    ------
    ValueError
        If the dates are invalid or exceed ``max_days``.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Dates must be in YYYY-MM-DD format") from exc

    if start >= end:
        raise ValueError("start_date must be earlier than end_date")

    if (end - start).days > max_days:
        raise ValueError(f"Date range cannot exceed {max_days} days")
