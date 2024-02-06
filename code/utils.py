import time
from contextlib import contextmanager


@contextmanager
def timing(label: str):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{label}: {end_time - start_time} seconds")
