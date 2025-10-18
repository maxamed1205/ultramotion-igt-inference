import time
import numpy as np

from core.queues.buffers import init_queues, get_queue_rt_dyn, get_queue_gpu, get_queue_out, collect_queue_metrics, enqueue_nowait_rt, apply_rt_backpressure
from core.preprocessing.cpu_to_gpu import process_one_frame
from core.inference.segmentation_engine import process_inference_once
from core.output.slicer_sender import start_sending_thread
from core.types import FrameMeta, RawFrame


def test_pipeline_end_to_end():
    init_queues({})
    # injecte une RawFrame simulée
    rf = RawFrame(image=np.zeros((10, 10), dtype=np.uint8), meta=FrameMeta(frame_id=1, ts=time.time()))
    # insert into RT queue using the public buffers helpers (non-blocking + backpressure)
    q = get_queue_rt_dyn()
    if not enqueue_nowait_rt(q, rf):
        apply_rt_backpressure(q, now=time.time(), max_lag_ms=500)
        enqueue_nowait_rt(q, rf)
    # traverse le pipeline complet
    process_one_frame()
    process_inference_once()
    start_sending_thread(pyigtl_server=None)

    m = collect_queue_metrics()
    assert isinstance(m["Queue_RT_dyn"]["size"], int)
    assert isinstance(m["Queue_GPU"]["size"], int)
    assert isinstance(m["Queue_Out"]["size"], int)
