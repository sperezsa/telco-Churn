"""Time-window helpers shared by the model monitoring processes."""

from datetime import datetime, timedelta, timezone


def previous_completed_week(
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Return UTC boundaries for the previous complete Monday-to-Monday week."""
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("The monitoring clock must include a timezone.")
    current_time = current_time.astimezone(timezone.utc)
    this_monday = (current_time - timedelta(days=current_time.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return this_monday - timedelta(days=7), this_monday
