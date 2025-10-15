"""Skeleton inference service for OpenIGTLink.

This script is a minimal starting point: it connects as a client to a PlusServer
OpenIGTLink endpoint, subscribes to IMAGE messages, performs a mocked
inference step, and republishes a binary mask IMAGE under the device name
`BoneMask` for Slicer to consume.

Replace the mock_infer() with the real D-FINE -> MobileSAM pipeline later.
"""

import time
import csv
from pathlib import Path

from igthelper import IGTGateway

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)


def mock_infer(image):
    # Simulate some processing time (e.g., 100 ms)
    time.sleep(0.1)
    # For now return a dummy mask: same shape as input, filled zeros
    import numpy as np
    return (np.zeros_like(image) > 0).astype('uint8')


def main():
    # Configuration
    plusserver_host = "127.0.0.1"
    plusserver_port = 18944
    slicer_listen_port = 18945

    gw = IGTGateway(plusserver_host, plusserver_port, slicer_listen_port)
    gw.start()

    log_file = LOG_PATH / "latency_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "t_in_igt", "t_recv", "t_after_infer", "t_sent_mask", "dt_infer_ms"])

        try:
            for frame_id, (img, meta) in gw.image_generator():
                t_recv = time.time()
                t_in_igt = meta.get("timestamp", None)

                mask = mock_infer(img)
                t_after = time.time()

                gw.send_mask(mask, meta)
                t_sent = time.time()

                writer.writerow([frame_id, t_in_igt, t_recv, t_after, t_sent, (t_after - t_recv) * 1000.0])
                f.flush()

        except KeyboardInterrupt:
            print("Shutting down")
        finally:
            gw.stop()


if __name__ == "__main__":
    main()
