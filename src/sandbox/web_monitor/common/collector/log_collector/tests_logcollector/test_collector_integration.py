"""
test_collector_integration.py
-----------------------------
Test d'intégration complet du LogCollector.

Ce test :
 - crée des faux fichiers pipeline.log et kpi.log temporaires,
 - écrit des lignes simulant RX→PROC→TX et interstage GPU,
 - démarre le LogCollector (sans tailers réels),
 - injecte les lignes directement dans les queues,
 - vérifie la cohérence du snapshot final (latences, timeline, etc.).
"""

import os
import tempfile
import time
import pytest

from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector


@pytest.fixture
def fake_logs(tmp_path):
    """Crée deux fichiers logs factices (pipeline.log et kpi.log)."""
    pipeline_file = tmp_path / "pipeline.log"
    kpi_file = tmp_path / "kpi.log"

    # Simule une frame complète (rx → proc → tx)
    pipeline_lines = [
        "[2025-10-30 12:00:00,000] [INFO] [DATASET-RX] Frame #001\n",
        "[2025-10-30 12:00:00,010] [INFO] [PROC-SIM] Processing frame #001\n",
        "[2025-10-30 12:00:00,020] [INFO] [TX-SIM] Sent frame #001\n",
    ]
    pipeline_file.write_text("".join(pipeline_lines), encoding="utf-8")

    # Simule des mesures GPU (interstage + copy_async)
    kpi_lines = [
        "kpi ts_log=2025-10-30 12:00:00,025 event=copy_async norm_ms=0.1 pin_ms=0.2 copy_ms=0.3 total_ms=0.6 frame=1\n",
        "[PROC-SIM]  Inter-stage latencies #001:\n"
        "  RX → CPU-to-GPU:    1.90ms\n"
        "  CPU-to-GPU → PROC:  10.83ms\n"
        "  PROC → GPU-to-CPU:  1.13ms\n"
        "  Total processing:   13.86ms | cuda\n",
    ]
    kpi_file.write_text("".join(kpi_lines), encoding="utf-8")

    return str(pipeline_file), str(kpi_file)


def test_collector_pipeline_cycle(fake_logs):
    """Test principal : vérifie que la fusion et le snapshot fonctionnent."""
    pipeline_path, kpi_path = fake_logs
    collector = LogCollector(pipeline_path, kpi_path)

    # Injection manuelle des lignes (pas de thread tailer)
    with open(pipeline_path, "r", encoding="utf-8") as f:
        for line in f:
            collector.q_pipeline.put(line)
    with open(kpi_path, "r", encoding="utf-8") as f:
        for line in f:
            collector.q_kpi.put(line)

    # Lancement manuel de la boucle de consommation
    collector._consume_loop()

    snap = collector.snapshot()
    data = snap.to_dict()

    # Vérifications de base
    assert data["health"] == "OK"
    assert data["latest"] is not None

    latest = data["latest"]
    lat = latest["interstage"]

    # Vérifie les latences GPU-résidentes
    assert round(lat["rx_cpu"], 2) == 1.90
    assert round(lat["cpu_gpu"], 2) == 10.83
    assert round(lat["proc_gpu"], 2) == 1.13
    assert round(lat["total"], 2) == 13.86

    # Vérifie que la timeline contient bien la frame
    timeline = data["timeline"]
    assert timeline["frames"] == [1]
    assert pytest.approx(timeline["total_ms"][0], 0.01) == 13.86

    # Vérifie cohérence GPU transfer
    gpu_transfer = latest["gpu_transfer"]
    assert round(gpu_transfer["total_ms"], 1) == 0.6
    assert round(gpu_transfer["pin_ms"], 1) == 0.2

    print("\n✅ Test LogCollector OK : Frame #001 agrégée correctement.")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
