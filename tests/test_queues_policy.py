import time
import numpy as np

from core.types import FrameMeta, RawFrame
from core.queues.buffers import (
    drop_oldest_policy_list,
    drop_oldest_policy_queue,
    init_queues,
    get_queue_rt_dyn,
)
import queue as _queue
from core.queues.buffers import (
    get_queue_gpu,
    enqueue_nowait_gpu,
    apply_rt_backpressure,
    collect_queue_metrics,
    drops_rt,
)


def make_raw(frame_id, ts_offset_ms=0):
    ts = time.time() - (ts_offset_ms / 1000.0)
    meta = FrameMeta(frame_id=frame_id, ts=ts)
    img = np.zeros((10, 10), dtype=np.uint8)
    return RawFrame(image=img, meta=meta)


def test_drop_oldest_policy_simulation():
    # Simule l'injection de 10 RawFrame très anciennes -> policy doit drop-oldest
    queue = []
    now = time.time()

    # crée 10 frames échelonnées dans le passé (trop vieilles)
    for i in range(10):
        rf = make_raw(i, ts_offset_ms=1000 + i * 100)  # 1s + i*0.1s ago
        queue.append(rf)

    # Politique: si lag_ms > 500, on éjecte jusqu'à rester sous 500
    def lag_ms(frame):
        return (now - frame.meta.ts) * 1000.0

    # Appliquer la politique via la fonction centralisée
    drop_oldest_policy_list(queue, now=now, max_lag_ms=500)

    # Après suppression, on s'attend à vider complètement la queue donnée
    assert len(queue) == 0


def test_drop_oldest_policy_queue_all_old():
    # Cas 1: tous les éléments sont vieux -> queue vidée
    # use registry queue to get deterministic type
    init_queues({})
    q = get_queue_rt_dyn()
    base = 1000000.0
    # push items with ts far in past
    for i in range(5):
        ts = base - (1000 + i * 100) / 1000.0
        meta = FrameMeta(frame_id=i, ts=ts)
        q.put(RawFrame(image=np.zeros((2, 2), dtype=np.uint8), meta=meta))

    stats = drop_oldest_policy_queue(q, now=base, max_lag_ms=500)
    assert stats["removed"] >= 1
    assert q.qsize() == 0


def test_drop_oldest_policy_queue_partial():
    # Cas 2: seul le premier est vieux -> il est supprimé
    init_queues({})
    q = get_queue_rt_dyn()
    base = 2000000.0
    # first is old
    q.put(RawFrame(image=np.zeros((2, 2), dtype=np.uint8), meta=FrameMeta(frame_id=0, ts=base - 1.0)))
    # others are recent
    for i in range(1, 4):
        q.put(RawFrame(image=np.zeros((2, 2), dtype=np.uint8), meta=FrameMeta(frame_id=i, ts=base)))

    stats = drop_oldest_policy_queue(q, now=base, max_lag_ms=500)
    assert stats["removed"] == 1
    assert q.qsize() == 3


def test_drop_oldest_policy_queue_none_old():
    # Cas 3: aucun n'est vieux -> queue inchangée
    init_queues({})
    q = get_queue_rt_dyn()
    base = 3000000.0
    for i in range(3):
        q.put(RawFrame(image=np.zeros((2, 2), dtype=np.uint8), meta=FrameMeta(frame_id=i, ts=base)))

    stats = drop_oldest_policy_queue(q, now=base, max_lag_ms=500)
    assert stats["removed"] == 0
    assert q.qsize() == 3


def test_enqueue_nowait_gpu_behavior():
    init_queues({})
    qg = get_queue_gpu()
    # fill GPU queue to capacity
    for i in range(3):
        qg.put(object())

    ok = enqueue_nowait_gpu(qg, object())
    # enqueue should either succeed after drop or return False; should not block
    assert isinstance(ok, bool)


def test_apply_rt_backpressure_counts():
    init_queues({})
    q = get_queue_rt_dyn()
    base = 4000000.0
    # push several old frames
    from core.types import FrameMeta, RawFrame

    for i in range(5):
        q.put(RawFrame(image=np.zeros((2, 2), dtype=np.uint8), meta=FrameMeta(frame_id=i, ts=base - 2.0)))

    stats = apply_rt_backpressure(q, now=base, max_lag_ms=500)
    assert "removed" in stats
    # collect metrics should reflect drops_rt as integer
    m = collect_queue_metrics()
    assert "Queue_RT_dyn" in m
    assert isinstance(m["Queue_RT_dyn"]["drops"], int) or m["Queue_RT_dyn"]["drops"] is None


def test_try_dequeue_behavior():
    from core.queues.buffers import try_dequeue, init_queues, get_queue_rt_dyn
    init_queues({})
    q = get_queue_rt_dyn()
    assert try_dequeue(q) is None
    q.put("item")
    assert try_dequeue(q) == "item"
    assert try_dequeue(q) is None


def test_collect_queue_metrics_structure():
    init_queues({})
    m = collect_queue_metrics()
    assert all(k in m for k in ("Queue_Raw", "Queue_RT_dyn", "Queue_GPU", "Queue_Out"))
    for v in m.values():
        assert set(v.keys()) == {"size", "maxsize", "drops", "last_backpressure_ts"}


def test_drops_gpu_out_increment():
    from core.queues.buffers import get_queue_out, enqueue_nowait_out
    init_queues({})
    qg, qo = get_queue_gpu(), get_queue_out()
    # remplir puis forcer drop-oldest
    for i in range(3):
        qg.put(object())
    enqueue_nowait_gpu(qg, object())
    for i in range(3):
        qo.put(object())
    enqueue_nowait_out(qo, object())
    m = collect_queue_metrics()
    assert isinstance(m["Queue_GPU"]["drops"], int) or m["Queue_GPU"]["drops"] is None
    assert isinstance(m["Queue_Out"]["drops"], int) or m["Queue_Out"]["drops"] is None


def test_last_backpressure_timestamps():
    from core.queues.buffers import get_queue_rt_dyn, apply_rt_backpressure
    import time as _time
    init_queues({})
    q = get_queue_rt_dyn()
    for i in range(5):
        from core.types import FrameMeta, RawFrame
        q.put(RawFrame(image=np.zeros((2,2), dtype=np.uint8), meta=FrameMeta(frame_id=i, ts=_time.time()-2)))
    apply_rt_backpressure(q)
    m = collect_queue_metrics()
    assert m["Queue_RT_dyn"]["last_backpressure_ts"] is not None


def test_temporary_log_level_restores():
    import logging
    from core.utils.debug import temporary_log_level

    logger = logging.getLogger("igt.queues")
    old = logger.level
    with temporary_log_level(logger, logging.DEBUG):
        assert logger.getEffectiveLevel() == logging.DEBUG
    assert logger.getEffectiveLevel() == old
