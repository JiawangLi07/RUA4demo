"""Confirm asynchronous Sharpa transitions without repeating the setter."""
import time


def await_enable_state(read, expected, check, timeout=.5):
    start = time.monotonic()
    consecutive = 0
    while time.monotonic() - start < timeout:
        value = read()
        check()
        consecutive = consecutive + 1 if value == expected else 0
        # A pending enable may still be in flight when cleanup requests disable.
        if consecutive >= 3 and (expected or time.monotonic() - start >= .12):
            return
        time.sleep(.015)
    raise RuntimeError(f"Sharpa enable state did not stabilize at {expected}")
