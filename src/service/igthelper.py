"""Minimal helper for OpenIGTLink I/O using pyigtl.

This file contains a very small wrapper class `IGTGateway` that:
- connects as a client to a PlusServer OpenIGTLink endpoint to receive IMAGE
  messages,
- opens a server socket for Slicer to connect and receive the published masks.

Note: This is a lightweight skeleton for development. Replace/expand with
robust error handling and full message parsing for production.
"""

import threading
import time
import numpy as np

try:
    import pyigtl
except Exception:
    pyigtl = None


class IGTGateway:
    def __init__(self, plus_host: str, plus_port: int, slicer_port: int):
        self.plus_host = plus_host
        self.plus_port = plus_port
        self.slicer_port = slicer_port
        self._running = False
        self._frame_counter = 0

    def start(self):
        # Start client/server connections (mock if pyigtl not available)
        self._running = True
        print(f"IGTGateway starting: plus={self.plus_host}:{self.plus_port} slicer_listen={self.slicer_port}")

    def stop(self):
        self._running = False
        print("IGTGateway stopped")

    def image_generator(self):
        """Yield tuple (frame_id, (image_array, meta_dict)).

        This mock generator produces a grayscale image every 0.05s. Replace with
        actual pyigtl IMAGE message parsing when pyigtl is installed.
        """
        while self._running:
            self._frame_counter += 1
            img = (np.random.rand(480, 640) * 255).astype('uint8')
            meta = {"timestamp": time.time()}
            yield self._frame_counter, (img, meta)
            time.sleep(0.05)

    def send_mask(self, mask_array: np.ndarray, meta: dict):
        # In a real implementation this will construct an IGT IMAGE message
        # and send it on the server socket to any connected Slicer clients.
        print(f"send_mask called — mask shape: {mask_array.shape}")
