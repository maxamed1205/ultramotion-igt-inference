"""
tailer.py
----------
Thread qui suit un fichier en continu, façon `tail -f`,
et met chaque nouvelle ligne dans une Queue.
"""

import time
import threading

from . import logger

class FileTailer(threading.Thread):
    def __init__(self, filepath, queue, poll_interval=0.05):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.queue = queue
        self.poll_interval = poll_interval
        self.running = False

    def run(self):
        self.running = True
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)  # skip to EOF
            while self.running:
                line = f.readline()
                if line:
                    self.queue.put(line)
                else:
                    time.sleep(self.poll_interval)

    def stop(self):
        self.running = False
