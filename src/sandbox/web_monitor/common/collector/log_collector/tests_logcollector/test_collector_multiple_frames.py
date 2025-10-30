"""
test_collector_multiple_frames.py
---------------------------------
Test d'intégration multi-frame du LogCollector.

Ce test :
 - simule plusieurs frames consécutives (#1, #2, #3),
 - injecte les logs pipeline.log et kpi.log,
 - vérifie la cohérence de la timeline,
 - contrôle la progression du snapshot et des latences.
"""

import pytest
from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector


@pytest.fixture
def fake_multi_logs(tmp_path):
    """Crée des logs factices avec 3 frames complètes (#1 → #3)."""
    pipeline_file = tmp_path / "pipeline.log"
    kpi_file = tmp_path / "kpi.log"

    pipeline_lines = []
    kpi_lines = []

    for fid in range(1, 4):
        # Pipeline RX→PROC→TX
        pipeline_lines += [
            f"[2025-10-30 12:00:0{fid},000] [INFO] [DATASET-RX] Frame #{fid:03d}\n",
            f"[2025-10-30 12:00:0{fid},010] [INFO] [PROC-SIM] Processing frame #{fid:03d}\n",
            f"[2025-10-30 12:00:0{fid},020] [INFO] [TX-SIM] Sent frame #{fid:03d}\n",
        ]

        # GPU metrics — varient légèrement à chaque frame
        rx_cpu = 1.90 + fid * 0.1
        cpu_gpu = 10.83 + fid * 0.05
        proc_gpu = 1.13 + fid * 0.02
        total = round(rx_cpu + cpu_gpu + proc_gpu, 2)
        copy_async_total = 0.6 + 0.1 * fid

        kpi_lines += [
            f"kpi ts_log=2025-10-30 12:00:0{fid},025 event=copy_async "
            f"norm_ms=0.1 pin_ms=0.2 copy_ms=0.3 total_ms={copy_async_total} frame={fid}\n",
            f"[PROC-SIM]  Inter-stage latencies #{fid:03d}:\n"
            f"  RX → CPU-to-GPU:    {rx_cpu:.2f}ms\n"
            f"  CPU-to-GPU → PROC:  {cpu_gpu:.2f}ms\n"
            f"  PROC → GPU-to-CPU:  {proc_gpu:.2f}ms\n"
            f"  Total processing:   {total:.2f}ms | cuda\n",
        ]

    pipeline_file.write_text("".join(pipeline_lines), encoding="utf-8")
    kpi_file.write_text("".join(kpi_lines), encoding="utf-8")

    return str(pipeline_file), str(kpi_file)


def test_collector_multiple_frames(fake_multi_logs):
    """Test principal multi-frame."""
    pipeline_path, kpi_path = fake_multi_logs
    collector = LogCollector(pipeline_path, kpi_path)

    # Injection manuelle
    with open(pipeline_path, "r", encoding="utf-8") as f:
        for line in f:
            collector.q_pipeline.put(line)
    with open(kpi_path, "r", encoding="utf-8") as f:
        for line in f:
            collector.q_kpi.put(line)

    # Traitement
    collector._consume_loop()
    snap = collector.snapshot()
    data = snap.to_dict()

    pretty_print_snapshot(snap)

    # Vérifications globales
    assert data["health"] == "OK"
    timeline = data["timeline"]
    frames = timeline["frames"]
    totals = timeline["total_ms"]

    assert frames == [1, 2, 3], "Les frames doivent être dans l'ordre chronologique."
    assert len(totals) == 3

    # Vérifie l'évolution des totaux
    assert totals[0] < totals[1] < totals[2], "Les latences totales doivent augmenter légèrement."

    # Vérifie la dernière frame (#3)
    latest = data["latest"]
    lat = latest["interstage"]
    gpu_transfer = latest["gpu_transfer"]

    assert round(lat["total"], 2) == round(sum([1.90 + 0.3, 10.83 + 0.15, 1.13 + 0.06]), 2)
    assert round(gpu_transfer["total_ms"], 1) == 0.9

    print("\n✅ Test LogCollector multi-frame OK : 3 frames agrégées et cohérentes.")

def pretty_print_snapshot(snap):
    """Affichage lisible du contenu d’un DashboardSnapshotLite."""
    data = snap.to_dict()
    latest = data.get("latest", {})
    timeline = data.get("timeline", {})

    print("\n" + "─" * 80)
    print(f"📊  DASHBOARD SNAPSHOT — Synthèse Collector ({data.get('health')})")
    print("─" * 80)

    # ---- Dernière frame ----
    if latest:
        fid = latest["frame_id"]
        print(f"🟩 Dernière Frame Agrégée  : #{fid}")
        inter = latest.get("interstage")
        gpu = latest.get("gpu_transfer")
        latcpu = latest.get("latency_rxtx")
        print("  ── CPU Path:")
        if latcpu:
            print(f"     RX→TX total      : {latcpu:6.2f} ms")
        print("  ── GPU Inter-stage:")
        if inter:
            for k, v in inter.items():
                if k not in ("ts_log", "ts_iso", "frame_id") and v is not None:
                    print(f"     {k:<10}: {v:6.2f} ms")
        print("  ── GPU Transfer:")
        if gpu:
            for k, v in gpu.items():
                if k not in ("frame_id", "ts_log") and v is not None:
                    print(f"     {k:<10}: {v:6.2f} ms")

    # ---- Timeline ----
    print("\n📈  Timeline (dernières frames)")
    frames = timeline.get("frames", [])
    totals = timeline.get("total_ms", [])
    if frames:
        for i, fid in enumerate(frames):
            tot = totals[i] if i < len(totals) else None
            print(f"   Frame #{fid:<3d} | Total GPU = {tot:6.2f} ms")

    print("─" * 80 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
