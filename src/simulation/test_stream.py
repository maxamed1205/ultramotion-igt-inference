"""Test unitaire simple pour vérifier le comportement du MockIGTGateway.

Ce script lance la simulation, itère sur quelques images et affiche la
dimension, l'index et le timestamp de chaque image. Usage:

    python -m src.simulation.test_stream

"""

import time
import logging
from simulation.mock_gateway import MockIGTGateway

LOG = logging.getLogger("igt.simulation")


def main() -> None:
    gateway = MockIGTGateway()
    gateway.start()

    try:
        gen = gateway.generate_images(interval_s=0.05)
        for i in range(10):
            img, meta = next(gen)
            LOG.info("Frame %d: shape=%s, ts=%.3f, id=%s", i, img.shape, meta["timestamp"], meta["frame_id"])
    except StopIteration:
        pass
    finally:
        gateway.stop()


if __name__ == '__main__':
    main()
