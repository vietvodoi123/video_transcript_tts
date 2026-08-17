def srt_time_to_seconds(
    value: str,
) -> float:

    hour, minute, second = value.split(":")

    second, ms = second.split(",")

    return (
        int(hour) * 3600
        + int(minute) * 60
        + int(second)
        + int(ms) / 1000
    )


def seconds_to_srt_time(
    seconds: float,
) -> str:

    total_ms = int(round(seconds * 1000))

    ms = total_ms % 1000
    total_seconds = total_ms // 1000

    sec = total_seconds % 60
    total_minutes = total_seconds // 60

    minute = total_minutes % 60
    hour = total_minutes // 60

    return (
        f"{hour:02d}:"
        f"{minute:02d}:"
        f"{sec:02d},"
        f"{ms:03d}"
    )