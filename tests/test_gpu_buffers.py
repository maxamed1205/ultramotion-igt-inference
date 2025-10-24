import pytest
import torch
from pprint import pprint
from core.queues.gpu_buffers import (
    init_gpu_pool,
    acquire_buffer,
    release_buffer,
    clear_gpu_pool,
    collect_gpu_metrics,
)

def test_gpu_pool_lifecycle():
    print("\n🧩 [1] Initialisation du pool GPU")
    init_gpu_pool(device="cuda:0" if torch.cuda.is_available() else "cpu")
    m1 = collect_gpu_metrics()
    pprint(m1)
    assert "_summary" in m1
    assert len(m1) >= 2  # summary + au moins 1 buffer

    print("\n🚀 [2] Acquisition d’un buffer 'roi'")
    roi = acquire_buffer("roi")
    assert roi is not None
    assert isinstance(roi, torch.Tensor)
    m2 = collect_gpu_metrics()
    pprint(m2)
    assert m2["roi"]["in_use"] is True

    print("\n♻️ [3] Libération du buffer 'roi'")
    release_buffer("roi")
    m3 = collect_gpu_metrics()
    pprint(m3)
    assert m3["roi"]["in_use"] is False

    print("\n🧹 [4] Nettoyage complet du pool GPU")
    clear_gpu_pool()
    m4 = collect_gpu_metrics()
    pprint(m4)
    assert m4 == {}  # pool vide après cleanup
