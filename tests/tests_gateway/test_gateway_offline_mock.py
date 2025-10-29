"""
Test offline : simulation complète de la Gateway (RX -> PROC -> TX) sans PlusServer ni GPU.

Ce test valide :
 - la communication entre threads via AdaptiveDeque,
 - la génération de frames fictives (RX),
 - le traitement simulé (PROC),
 - et la transmission simulée (TX),
avec enregistrement complet dans le système de logs asynchrone.
"""

import sys
from pathlib import Path

# ──────────────────────────────────────────────
#  Correction console Windows : forcer UTF-8
# ──────────────────────────────────────────────
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
#  CHEMINS ET IMPORTS DE BASE
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import time
import threading
import numpy as np
from core.types import FrameMeta, RawFrame, ResultPacket
from core.queues.adaptive import AdaptiveDeque

# ──────────────────────────────────────────────
#  INITIALISATION DU LOGGING ASYNCHRONE
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
    print("⚠️ logging.yaml non trouvé → logging désactivé")

# Création des loggers spécifiques à chaque thread
LOG_RX = logging.getLogger("igt.receiver")    # thread de réception
LOG_PROC = logging.getLogger("igt.inference") # thread de traitement
LOG_TX = logging.getLogger("igt.slicer")      # thread d’envoi (Slicer)


# ---------------------------------------------------------------------
# 1️⃣ Simulateur RX : crée des images et poses fictives
# ---------------------------------------------------------------------
def simulate_rx(mailbox: AdaptiveDeque, stop_event: threading.Event, fps: int = 10):
    """Génère des RawFrame fictifs à ~fps Hz et les insère dans _mailbox."""
    frame_id = 0
    LOG_RX.info("[RX] Thread started (simulating incoming frames)")
    while not stop_event.is_set():
        img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        meta = FrameMeta(frame_id=frame_id, ts=time.time())
        frame = RawFrame(image=img, meta=meta)
        mailbox.append(frame)
        LOG_RX.debug(f"[RX] Produced frame #{frame_id:03d} (mailbox size={len(mailbox)})")
        frame_id += 1
        time.sleep(1.0 / fps)
    LOG_RX.info("[RX] Thread stopped.")


# ---------------------------------------------------------------------
# 2️⃣ Traitement fictif : lit _mailbox et écrit _outbox
# ---------------------------------------------------------------------
def simulate_processing(mailbox: AdaptiveDeque, outbox: AdaptiveDeque, stop_event: threading.Event):
    """Lit les RawFrame et les transforme en ResultPacket simulé."""
    from core.monitoring.kpi import safe_log_kpi, format_kpi

    LOG_PROC.info("[PROC] Thread started (mock inference)")
    while not stop_event.is_set():
        if len(mailbox) == 0:
            time.sleep(0.01)
            continue
        try:
            rf = mailbox.pop()
            mask = (rf.image > 128).astype(np.uint8)
            result = ResultPacket(mask=mask, score=0.9, state="VISIBLE", meta=rf.meta)
            outbox.append(result)

            kmsg = format_kpi({
                "ts": time.time(),
                "event": "mock_process",
                "frame_id": rf.meta.frame_id,
                "score": 0.9,
            })
            safe_log_kpi(kmsg)

            LOG_PROC.info(f"[PROC] Processed frame #{rf.meta.frame_id:03d} -> outbox size={len(outbox)}")
        except Exception as e:
            LOG_PROC.exception(f"[PROC] Processing error: {e}")
    LOG_PROC.info("[PROC] Thread stopped.")


# ---------------------------------------------------------------------
# 3️⃣ Simulateur TX : lit _outbox et “envoie” les résultats
# ---------------------------------------------------------------------
def simulate_tx(outbox: AdaptiveDeque, stop_event: threading.Event):
    """Lit les résultats et les affiche (simule l’envoi Slicer)."""
    LOG_TX.info("[TX] Thread started (simulating sending to Slicer)")
    while not stop_event.is_set():
        if len(outbox) == 0:
            time.sleep(0.05)
            continue
        pkt = outbox.popleft()
        fid = pkt.meta.frame_id if pkt.meta else -1
        LOG_TX.info(f"[TX] Sent {pkt.meta.device_name}#{fid:03d} (state={pkt.state}, score={pkt.score:.2f}, mask={pkt.mask.shape})")
    LOG_TX.info("[TX] Thread stopped.")


# ---------------------------------------------------------------------
# 4️⃣ Lancement de la simulation
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mailbox = AdaptiveDeque(maxlen=8)
    outbox = AdaptiveDeque(maxlen=8)
    stop_event = threading.Event()

    rx_thread = threading.Thread(target=simulate_rx, args=(mailbox, stop_event), daemon=True)
    proc_thread = threading.Thread(target=simulate_processing, args=(mailbox, outbox, stop_event), daemon=True)
    tx_thread = threading.Thread(target=simulate_tx, args=(outbox, stop_event), daemon=True)

    rx_thread.start()
    proc_thread.start()
    tx_thread.start()

    time.sleep(5)
    stop_event.set()
    time.sleep(1)

    print("✅ Test offline terminé (voir logs/pipeline.log et logs/kpi.log pour les détails).")
