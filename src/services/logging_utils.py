from typing import Callable, Optional


def print_with_prefix(prefix: str, message: Optional[str], enabled: bool = True) -> None:
    if not enabled:
        return
    text = "" if message is None else str(message)
    lines = text.splitlines() or [""]
    for line in lines:
        if line:
            print(f"{prefix} {line}")
        else:
            print(prefix)


def log_section(
    log_fn: Callable[[str], None],
    title: str,
    width: int = 70,
    char: str = "=",
) -> None:
    line = char * width
    log_fn(line)
    log_fn(title)
    log_fn(line)
