"""
Test de pipeline continue pour alimenter le dashboard
======================================================

Lance une simulation continue de la pipeline pour générer des métriques.
Laissez tourner ce script pendant que vous consultez le dashboard.

Usage:
    python tests/tests_gateway/test_continuous_for_dashboard.py
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Configure logging
import logging.config, yaml
from core.monitoring import async_logging

LOG_CFG = ROOT / "src" / "config" / "logging.yaml"
if LOG_CFG.exists():
    with open(LOG_CFG, "r") as f:
        cfg = yaml.safe_load(f)
        async_logging.setup_async_logging(yaml_cfg=cfg)
        async_logging.start_health_monitor()

# Imports
from service.gateway.manager import IGTGateway
from service.slicer_server import run_slicer_server
from core.types import RawFrame, FrameMeta, Pose

LOG = logging.getLogger("igt.gateway.continuous")

# ═════════════════════════════════════════════════════════════
# SIMULATION CONTINUE
# ═════════════════════════════════════════════════════════════

def simulate_frame_source(gateway: IGTGateway, stop_event: threading.Event, fps: int = 30):
    """Génère des frames en continu"""
    frame_id = 0
    interval = 1.0 / fps
    LOG.info(f"[RX-CONTINUOUS] Frame generator started at {fps} Hz")
    
    while not stop_event.is_set():
        # Générer une image simulée
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
            device_name="SimulatedUS",
        )
        
        frame = RawFrame(image=img, meta=meta)
        gateway._inject_frame(frame)
        
        frame_id += 1
        
        # Log toutes les 100 frames
        if frame_id % 100 == 0:
            LOG.info(f"[RX-CONTINUOUS] Generated {frame_id} frames")
        
        time.sleep(interval)
    
    LOG.info(f"[RX-CONTINUOUS] Stopped after {frame_id} frames")


def simulate_processing(gateway: IGTGateway, stop_event: threading.Event):
    """Traitement simulé (seuillage simple)"""
    processed = 0
    LOG.info("[PROC-CONTINUOUS] Processing thread started")
    
    while not stop_event.is_set():
        frame = gateway._get_next_frame()
        if frame is None:
            time.sleep(0.001)
            continue
        
        # Simulation de traitement (seuillage)
        mask = (frame.image > 128).astype(np.uint8) * 255
        
        # Envoyer vers la queue de sortie
        gateway._push_result(frame.meta.frame_id, mask)
        
        processed += 1
        
        if processed % 100 == 0:
            LOG.info(f"[PROC-CONTINUOUS] Processed {processed} frames")
    
    LOG.info(f"[PROC-CONTINUOUS] Stopped after {processed} frames")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    """Lance la simulation continue"""
    
    print("=" * 60)
    print("  PIPELINE CONTINUE POUR DASHBOARD")
    print("=" * 60)
    print()
    print("  Cette pipeline tourne en continu pour alimenter le dashboard.")
    print("  Lancez le dashboard dans un autre terminal avec:")
    print()
    print("      .\\start_dashboard.ps1")
    print()
    print("  Puis ouvrez http://localhost:8050")
    print()
    print("  Appuyez sur Ctrl+C pour arrêter.")
    print("=" * 60)
    print()
    
    LOG.info("Initialisation du IGTGateway (mode simulation continue)")
    
    # Créer le gateway
    gateway = IGTGateway(
        plus_server_address="127.0.0.1",
        plus_server_port=18944,
        slicer_port=18945,
        fps=30.0,
        interval=2.0
    )
    
    # Events de contrôle
    stop_event = threading.Event()
    
    # Threads
    rx_thread = threading.Thread(
        target=simulate_frame_source,
        args=(gateway, stop_event, 30),
        name="RX-Continuous",
        daemon=True
    )
    
    proc_thread = threading.Thread(
        target=simulate_processing,
        args=(gateway, stop_event),
        name="PROC-Continuous",
        daemon=True
    )
    
    tx_thread = threading.Thread(
        target=run_slicer_server,
        args=(gateway._queue_out, gateway._slicer_port),
        name="TX-Continuous",
        daemon=True
    )
    
    # Démarrage
    rx_thread.start()
    proc_thread.start()
    tx_thread.start()
    
    LOG.info("Pipeline démarrée - Génération de métriques en cours...")
    print("\n✅ Pipeline active ! Consultez le dashboard sur http://localhost:8050\n")
    
    try:
        # Boucle infinie jusqu'à Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt demandé par l'utilisateur...")
        LOG.info("Arrêt de la pipeline...")
        stop_event.set()
        
        # Attendre que les threads se terminent
        time.sleep(2)
        
        print("✅ Pipeline arrêtée proprement\n")
        LOG.info("Pipeline arrêtée")


if __name__ == "__main__":
    main()
