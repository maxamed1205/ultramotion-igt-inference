"""
Test offline : simulation complète de la vraie pipeline Ultramotion (GatewayManager)
avec génération de frames simulées à 100 Hz (toutes les 10 ms),
traitement local (seuillage simple) et envoi simulé via run_slicer_server().
"""

# ──────────────────────────────────────────────
#  Optimisation NumPy : limiter à 1 thread OMP
# ──────────────────────────────────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
def simulate_frame_source(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,  # ← Nouveau paramètre pour signaler frames disponibles
    fps: int = 100
):
    """Génère des RawFrame toutes les 10 ms et les injecte dans la vraie pipeline."""
    frame_id = 0
    interval = 1.0 / fps
    LOG.info(f"[RX-SIM] Frame generator started at {fps} Hz")

    # ──────────────────────────────────────────────
    # Horloge compensée : garantit 10.0ms ±0.1ms
    # ──────────────────────────────────────────────
    next_frame_time = time.perf_counter()

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
        LOG.info(f"[RX-SIM] Generated frame #{frame_id:03d}")  # ← Log AVANT injection (timestamp = création)
        gateway._inject_frame(frame)
        frame_ready.set()  # ← Signal : "une frame est dispo !"
        frame_id += 1

        # ──────────────────────────────────────────────
        # Sleep compensé : attend jusqu'à next_frame_time
        # ──────────────────────────────────────────────
        next_frame_time += interval
        now = time.perf_counter()
        sleep_duration = next_frame_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        # Si on est en retard (sleep_duration < 0), on continue immédiatement

    LOG.info("[RX-SIM] Generator stopped.")

# ──────────────────────────────────────────────
#  Traitement PROC (seuillage)
# ──────────────────────────────────────────────
def simulate_processing(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event  # ← Nouveau paramètre pour attendre les frames
):
    """Lit la mailbox, applique un seuillage, envoie vers outbox via send_mask()."""
    LOG.info("[PROC-SIM] Thread started (simple thresholding)")
    while not stop_event.is_set():
        # Attendre qu'une frame soit disponible (timeout 10ms pour éviter blocage infini)
        if not frame_ready.wait(timeout=0.01):
            continue  # Timeout → revérifier stop_event
        frame_ready.clear()  # Reset l'event pour la prochaine frame
        
        try:
            frame = gateway.receive_image()
            if frame is None:
                continue
            
            LOG.info(f"[PROC-SIM] Processing frame #{frame.meta.frame_id:03d}")  # ← Log AVANT traitement
            
            mask = (frame.image > 128).astype(np.uint8)
            meta = {
                "frame_id": frame.meta.frame_id,
                "ts": time.time(),
                "state": "VISIBLE",
            }
            gateway.send_mask(mask, meta)
            
            # 🔬 INSTRUMENTATION : Logger la taille de l'outbox après envoi
            outbox_size = len(gateway._outbox)
            if outbox_size > 0:
                LOG.debug(f"[PROC-SIM] Outbox size after send: {outbox_size}")
        except Exception as e:
            LOG.exception(f"[PROC-SIM] Error: {e}")
    LOG.info("[PROC-SIM] Thread stopped.")

# ──────────────────────────────────────────────
#  Test principal (RX → PROC → TX)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # OPTIMISATION 1/3 : Activer timer Windows 1ms
    # ──────────────────────────────────────────────
    from utils.win_timer_resolution import enable_high_resolution_timer
    enable_high_resolution_timer()
    
    LOG.info("Initialisation du IGTGateway (vraie pipeline, mode mock)")

    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    gateway._running = True  # mode offline

    stop_event = threading.Event()
    frame_ready = threading.Event()  # ← Signal quand une frame est disponible

    # Threads RX / PROC / TX (TX = run_slicer_server officiel)
    rx_thread = threading.Thread(
        target=simulate_frame_source,
        args=(gateway, stop_event, frame_ready),  # ← Ajouter frame_ready
        daemon=True
    )
    proc_thread = threading.Thread(
        target=simulate_processing,
        args=(gateway, stop_event, frame_ready),  # ← Ajouter frame_ready
        daemon=True
    )
    tx_thread = threading.Thread(
        target=run_slicer_server,
        args=(
            gateway._outbox,
            stop_event,
            18945,
            gateway.update_tx_stats,
            gateway.events.emit,
            gateway._tx_ready  # 🔬 OPTIMISATION : Passer l'Event pour réveil instantané
        ),
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
