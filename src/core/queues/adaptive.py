"""
core.queues.adaptive
--------------------

Adaptive queue utilities and a small AdaptiveDeque wrapper that supports
dynamic resizing of the logical capacity without changing caller references.
"""
from collections import deque
import logging
from threading import Lock
from typing import Tuple

LOG = logging.getLogger("igt.queues.adaptive")


class AdaptiveDeque:
    """A thin wrapper around collections.deque that allows dynamic maxlen.

    The wrapper preserves the same object identity so producers/consumers
    holding a reference to the AdaptiveDeque keep operating normally.
    """

    def __init__(self, maxlen: int = 8):
        self._lock = Lock()
        self._maxlen = int(maxlen)
        self._dq = deque(maxlen=self._maxlen)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def resize(self, new_maxlen: int) -> None:
        with self._lock:
            new_maxlen = int(new_maxlen)
            if new_maxlen == self._maxlen:
                return
            try:
                self._dq = deque(self._dq, maxlen=new_maxlen)
                self._maxlen = new_maxlen
                LOG.debug("AdaptiveDeque resized -> %d", new_maxlen)
            except Exception:
                LOG.exception("AdaptiveDeque resize failed")

    # Minimal deque interface used by the project
    def append(self, item):
        with self._lock:
            self._dq.append(item)

    def popleft(self):
        with self._lock:
            return self._dq.popleft()

    def pop(self):
        with self._lock:
            return self._dq.pop()

    def clear(self):
        with self._lock:
            self._dq.clear()

    def __len__(self):
        with self._lock:
            return len(self._dq)

    def __iter__(self):
        # iterate a snapshot to avoid locking during iteration
        with self._lock:
            return iter(list(self._dq))


def adjust_queue_size(q, fps_rx: float, fps_tx: float,
                      mb_rx: float, mb_tx: float,
                      min_len: int = 2, max_len: int = 16) -> Tuple[object, int]:
    """Adjust queue capacity based on observed metrics.

    Returns a tuple (q, new_len). For AdaptiveDeque the resize is in-place; for
    regular deque, a new deque object is returned along with new_len.
    """
    try:
        # tolerate either AdaptiveDeque or plain deque
        cur_len = getattr(q, "maxlen", None)
        if cur_len is None:
            # best-effort: treat as unbounded -> no change
            return q, cur_len

        ratio_fps = fps_rx / max(1e-3, fps_tx)
        ratio_mb = mb_rx / max(1e-3, mb_tx)
        new_len = cur_len

        if ratio_fps > 1.2 or ratio_mb > 1.2:
            new_len = min(max_len, cur_len + 1)
        elif ratio_fps < 0.8 and ratio_mb < 0.8:
            new_len = max(min_len, cur_len - 1)

        if new_len != cur_len:
            # perform resize
            if hasattr(q, "resize"):
                try:
                    q.resize(new_len)
                    return q, new_len
                except Exception:
                    LOG.exception("Adaptive resize via resize() failed")
                    return q, cur_len
            else:
                # replace plain deque with a new one
                try:
                    new_q = deque(q, maxlen=new_len)
                    LOG.debug("Queue resized adaptively → %d", new_len)
                    return new_q, new_len
                except Exception:
                    LOG.exception("Adaptive resize (plain deque) failed")
                    return q, cur_len

        return q, cur_len
    except Exception:
        LOG.exception("Adaptive resize failed")
        return q, getattr(q, "maxlen", None)
