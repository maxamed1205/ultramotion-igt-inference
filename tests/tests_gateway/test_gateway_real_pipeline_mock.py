"""
Test offline : simulation complète de la vraie pipeline Ultramotion (GatewayManager)
avec génération de frames simulées à 100 Hz (toutes les 10 ms),
traitement local (seuillage simple) et envoi simulé via run_slicer_server().
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
#  Correction console Windows : forcer UTF-8
# ──────────────────────────────────────────────
if sys.platform.startswith("win"):
    import io, os
    os.system("chcp 65001 >NUL")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
#  Préparation du contexte
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ──────────────────────────────────────────────
#  Imports pipeline réelle
# ──────────────────────────────────────────────
from service.gateway.manager import IGTGateway
from service.slicer_server import run_slicer_server
from core.types import RawFrame, FrameMeta, Pose

# ──────────────────────────────────────────────
#  Logger asynchrone
# ──────────────────────────────────────────────
import logging.config, yaml
from core.monitoring import async_logging

LOG_CFG = ROOT / "src" / "config" / "logging.yaml"
if LOG_CFG.exists():
    with open(LOG_CFG, "r") as f:
        cfg = yaml.safe_load(f)
        # ⚠️ Ne PAS appeler dictConfig avant setup_async_logging (évite duplication des handlers)
        # logging.config.dictConfig(cfg)  # <- COMMENTÉ : setup_async_logging gère déjà la configuration
        async_logging.setup_async_logging(yaml_cfg=cfg)
        async_logging.start_health_monitor()
else:
    print("logging.yaml non trouvé → logging désactivé")

LOG = logging.getLogger("igt.gateway.test")

# ──────────────────────────────────────────────
#  Simulateur RX (frames)
# ──────────────────────────────────────────────
def simulate_frame_source(gateway: IGTGateway, stop_event: threading.Event, fps: int = 100):
    """Génère des RawFrame toutes les 10 ms et les injecte dans la vraie pipeline."""
    frame_id = 0
    interval = 1.0 / fps
    LOG.info(f"[RX-SIM] Frame generator started at {fps} Hz")

    while not stop_event.is_set():
        img = (np.random.rand(512, 512) * 255).astype(np.uint8)
        pose = Pose()
        ts = time.time()
        meta = FrameMeta(
            frame_id=frame_id,
            ts=ts,
            pose=pose,
            spacing=(0.3, 0.3, 1.0),
            orientation="UN",
            coord_frame="Image",
            device_name="Image",
        )
        frame = RawFrame(image=img, meta=meta)
        gateway._inject_frame(frame)
        LOG.info(f"[RX-SIM] Generated frame #{frame_id:03d}")
        frame_id += 1
        time.sleep(interval)

    LOG.info("[RX-SIM] Generator stopped.")

# ──────────────────────────────────────────────
#  Traitement PROC (seuillage)
# ──────────────────────────────────────────────
def simulate_processing(gateway: IGTGateway, stop_event: threading.Event):
    """Lit la mailbox, applique un seuillage, envoie vers outbox via send_mask()."""
    LOG.info("[PROC-SIM] Thread started (simple thresholding)")
    while not stop_event.is_set():
        if len(gateway._mailbox) == 0:
            time.sleep(0.0005)
            continue
        try:
            frame = gateway.receive_image()
            if frame is None:
                continue
            mask = (frame.image > 128).astype(np.uint8)
            meta = {
                "frame_id": frame.meta.frame_id,
                "ts": time.time(),
                "state": "VISIBLE",
            }
            gateway.send_mask(mask, meta)
            LOG.info(f"[PROC-SIM] Processed frame #{frame.meta.frame_id:03d}")
        except Exception as e:
            LOG.exception(f"[PROC-SIM] Error: {e}")
    LOG.info("[PROC-SIM] Thread stopped.")

# ──────────────────────────────────────────────
#  Test principal (RX → PROC → TX)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    LOG.info("Initialisation du IGTGateway (vraie pipeline, mode mock)")

    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    gateway._running = True  # mode offline

    stop_event = threading.Event()

    # Threads RX / PROC / TX (TX = run_slicer_server officiel)
    rx_thread = threading.Thread(target=simulate_frame_source, args=(gateway, stop_event), daemon=True)
    proc_thread = threading.Thread(target=simulate_processing, args=(gateway, stop_event), daemon=True)
    tx_thread = threading.Thread(
        target=run_slicer_server,
        args=(gateway._outbox, stop_event, 18945, gateway.update_tx_stats, gateway.events.emit),
        daemon=True,
    )

    rx_thread.start()
    proc_thread.start()
    tx_thread.start()

    run_duration = 1.0
    LOG.info(f"Simulation en cours pendant {run_duration:.1f} s…")
    time.sleep(run_duration)

    stop_event.set()
    rx_thread.join(timeout=1.0)
    proc_thread.join(timeout=1.0)
    tx_thread.join(timeout=1.0)

    LOG.info(f"Simulation terminée — outbox restante: {len(gateway._outbox)}")
