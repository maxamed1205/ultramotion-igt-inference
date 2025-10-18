import time
from core.types import FrameMeta


class MockMsg:
    metadata = {"FrameNumber": 42, "Spacing": (0.5, 0.5, 1.0)}
    timestamp = 0.0
    device_name = "Image"


def run_benchmark(n: int = 100000):
    msg = MockMsg()
    start = time.time()
    for _ in range(n):
        FrameMeta.from_igt(msg)
    elapsed = time.time() - start
    print(f"{n} conversions: {elapsed:.4f} s")


if __name__ == "__main__":
    run_benchmark()
