"""
Debug / test hook: count how often roster JSON is read from disk (deprecated path).

Dashboard "C2 lineup" blocks should not increment this counter.
"""

json_file_read_count: int = 0


def track_json_file_read() -> None:
    global json_file_read_count
    json_file_read_count += 1


def reset_json_file_read_count() -> None:
    global json_file_read_count
    json_file_read_count = 0
