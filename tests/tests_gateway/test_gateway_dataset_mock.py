"""
Test offline : Pipeline Ultramotion avec VRAIES images du dataset
==================================================================

Étape 1 : Remplace la génération aléatoire d'images par la lecture séquentielle
du dataset JPEGImages/Video_001/ (213 images).

Conservation :
- ✅ Seuillage simple (simulate_processing)
- ✅ run_slicer_server (TX)
- ✅ Dashboard temps réel
- ✅ Cadence 100 Hz (10 ms entre frames)

Modification :
- 🔄 simulate_frame_source() → read_dataset_images()
  - Lit les vraies images JPEG
  - Les redimensionne en 512x512 (comme avant)
  - Cycle en boucle si durée > nombre d'images
"""

# ──────────────────────────────────────────────
#  Optimisation NumPy : limiter à 1 thread OMP
# ──────────────────────────────────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ✅ ACTIVER MODE DEBUG pour voir les logs de latence
os.environ["LOG_MODE"] = "dev"  # dev=INFO/DEBUG, perf=WARNING

import sys
import time
import threading
import signal
import numpy as np
from pathlib import Path
from PIL import Image
import glob

# ============================================================
# 🔧 UTF-8 SAFE MODE FOR WINDOWS CONSOLE
# (forces all logs and prints to use UTF-8 everywhere)
# ============================================================
import sys, io, os, locale

# Force UTF-8 code page for subprocesses
os.system("chcp 65001 >NUL")

# Force Python's stdout/stderr to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Confirm current locale (for debug)
print(f"[DEBUG] Console encoding: {sys.stdout.encoding}, locale: {locale.getpreferredencoding(False)}")


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
import torch  # ✅ Import torch pour les opérations GPU
from service.gateway.manager import IGTGateway
from service.slicer_server import run_slicer_server
from core.types import RawFrame, FrameMeta, Pose
from core.preprocessing.cpu_to_gpu import (
    init_transfer_runtime,
    prepare_frame_for_gpu,
)

# ──────────────────────────────────────────────
#  Logger asynchrone (sera configuré dans __main__)
# ──────────────────────────────────────────────
import logging.config, yaml
from core.monitoring import async_logging

LOG = logging.getLogger("igt.gateway.test")


# ──────────────────────────────────────────────
#  Nettoyage des logs avant test
# ──────────────────────────────────────────────
def clean_old_logs():
    """Supprime tous les fichiers .log dans le dossier logs/ avant de démarrer un nouveau test.
    
    Note: Doit être appelé AVANT setup_async_logging() pour éviter les conflits de verrouillage.
    """
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        return
    
    deleted_count = 0
    for log_file in logs_dir.glob("*.log"):
        try:
            log_file.unlink()
            deleted_count += 1
        except Exception as e:
            # Fichier verrouillé par un processus (logging actif) → ignorer
            pass
    
    if deleted_count > 0:
        print(f"[CLEAN] {deleted_count} fichier(s) log supprimé(s)")
    else:
        print("[CLEAN] Aucun log à supprimer (ou fichiers verrouillés)")


def setup_logging():
    """Configure le système de logging asynchrone."""
    LOG_CFG = ROOT / "src" / "config" / "logging.yaml"
    if LOG_CFG.exists():
        with open(LOG_CFG, "r") as f:
            cfg = yaml.safe_load(f)
            async_logging.setup_async_logging(yaml_cfg=cfg)
            async_logging.start_health_monitor()
    else:
        print("logging.yaml non trouve -> logging desactive")


def init_gpu_if_available():
    """Initialise le GPU si disponible, sinon utilise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            LOG.info(f"[OK] CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            LOG.info("[WARNING] CUDA non disponible, utilisation CPU")
        
        # Initialiser le runtime de transfert
        init_transfer_runtime(device=device, pool_size=2, shape_hint=(512, 512))
        LOG.info(f"[OK] Runtime GPU initialisé (device={device})")
        return device
    except ImportError:
        LOG.warning("[WARNING] PyTorch non installé, utilisation CPU seulement")
        return "cpu"
    except Exception as e:
        LOG.warning(f"[WARNING] Erreur GPU, fallback CPU: {e}")
        return "cpu"


# ──────────────────────────────────────────────
#  DATASET PATH (à adapter si nécessaire)
# ──────────────────────────────────────────────
DATASET_PATH = Path(r"C:\Users\maxam\Desktop\TM\dataset\HUMERUS LATERAL XG SW_cropped\JPEGImages\Video_001")


# ──────────────────────────────────────────────
#  Lecteur d'images réelles (remplace simulate_frame_source)
# ──────────────────────────────────────────────
def read_dataset_images(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,
    fps: int = 100,
    target_size: tuple = (512, 512),
    loop_mode: bool = False  # Nouveau paramètre : boucler ou s'arrêter après toutes les images
):
    """Lit séquentiellement les images du dataset et les injecte dans la pipeline.
    
    Args:
        gateway: Instance IGTGateway
        stop_event: Signal d'arrêt
        frame_ready: Signal de synchronisation avec PROC thread
        fps: Cadence d'envoi (Hz)
        target_size: Taille de redimensionnement (H, W)
        loop_mode: Si True, boucle indéfiniment. Si False, s'arrête après avoir envoyé toutes les images.
    """
    # ─────────────────────────────────────────────
    # 1. Charger la liste des images
    # ─────────────────────────────────────────────
    if not DATASET_PATH.exists():
        LOG.error(f"Dataset path not found: {DATASET_PATH}")
        LOG.error("Please update DATASET_PATH in the script")
        return
    
    image_files = sorted(glob.glob(str(DATASET_PATH / "*.jpg")))
    
    if not image_files:
        LOG.error(f"No JPEG images found in {DATASET_PATH}")
        return
    
    num_images = len(image_files)
    LOG.info(f"[DATASET-RX] Loaded {num_images} images from {DATASET_PATH.name}")
    LOG.info(f"[DATASET-RX] First image: {Path(image_files[0]).name}")
    LOG.info(f"[DATASET-RX] Last image: {Path(image_files[-1]).name}")
    LOG.info(f"[DATASET-RX] Target FPS: {fps} Hz (interval: {1000.0/fps:.1f} ms)")
    
    # ─────────────────────────────────────────────
    # 2. Boucle d'envoi à cadence régulière
    # ─────────────────────────────────────────────
    frame_id = 0
    image_idx = 0  # Index dans la liste d'images
    interval = 1.0 / fps
    next_frame_time = time.perf_counter()
    
    while not stop_event.is_set():
        # Lire l'image courante
        img_path = image_files[image_idx]
        
        try:
            # Charger et redimensionner l'image
            with Image.open(img_path) as pil_img:
                # Convertir en niveaux de gris (comme les images simulées étaient mono-canal)
                pil_img = pil_img.convert("L")  # Grayscale
                pil_img = pil_img.resize(target_size, Image.BILINEAR)
                img = np.array(pil_img, dtype=np.uint8)
            
            # Créer la frame avec métadonnées
            pose = Pose()
            ts = time.time()
            meta = FrameMeta(
                frame_id=frame_id,
                ts=ts,
                pose=pose,
                spacing=(0.3, 0.3, 1.0),
                orientation="UN",
                coord_frame="Image",
                device_name="Dataset",  # Identifier la source
            )
            frame = RawFrame(image=img, meta=meta)
            
            # Log toutes les 10 frames pour éviter spam
            if frame_id % 10 == 0:
                LOG.info(
                    f"[DATASET-RX] Frame #{frame_id:03d} | Image: {Path(img_path).name} | Shape: {img.shape}"
                )
            
            # Injection dans la pipeline
            gateway._inject_frame(frame)
            frame_ready.set()  # Signal pour PROC thread
            
            # ✅ INSTRUMENTATION : Enregistrer RX timestamp pour calcul de latence
            gateway.stats.mark_rx(frame_id, ts)
            
            # Avancer les compteurs
            frame_id += 1
            image_idx += 1
            
            # Si on a envoyé toutes les images
            if image_idx >= num_images:
                if loop_mode:
                    # Mode boucle : recommencer à 0
                    image_idx = 0
                    LOG.info(f"[DATASET-RX] Fin du dataset atteinte, redémarrage en boucle...")
                else:
                    # Mode une seule passe : arrêter
                    LOG.info(f"[DATASET-RX] Toutes les {num_images} images envoyées. Arrêt.")
                    break
            
        except Exception as e:
            LOG.error(f"[DATASET-RX] Failed to load {img_path}: {e}")
            # Passer à l'image suivante
            image_idx = (image_idx + 1) % num_images
            continue
        
        # ─────────────────────────────────────────────
        # Sleep compensé pour maintenir FPS constant
        # ─────────────────────────────────────────────
        next_frame_time += interval
        now = time.perf_counter()
        sleep_duration = next_frame_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        # Si en retard, continuer immédiatement
    
    LOG.info(f"[DATASET-RX] Stopped after {frame_id} frames")


# ──────────────────────────────────────────────
#  Traitement PROC (seuillage) - INCHANGÉ
# ──────────────────────────────────────────────
def simulate_processing(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,
    use_gpu: bool = False,
    gpu_device: str = "cpu"
):
    """Lit la mailbox, applique un seuillage (optionnellement sur GPU), envoie vers outbox via send_mask()."""
    proc_type = "GPU thresholding" if use_gpu else "simple thresholding"
    LOG.info(f"[PROC-SIM] Thread started ({proc_type}, device={gpu_device})")
    
    while not stop_event.is_set():
        # Attendre qu'une frame soit disponible (timeout 10ms pour éviter blocage infini)
        if not frame_ready.wait(timeout=0.01):
            continue  # Timeout → revérifier stop_event
        frame_ready.clear()  # Reset l'event pour la prochaine frame
        
        try:
            frame = gateway.receive_image()
            if frame is None:
                continue
            
            frame_id = frame.meta.frame_id
            LOG.info(f"[PROC-SIM] Processing frame #{frame_id:03d}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🎯 MÉTRIQUES INTER-ÉTAPES DÉTAILLÉES pour le workflow complet :
            # RX → CPU-to-GPU → PROC(GPU) → GPU-to-CPU → TX
            # ═══════════════════════════════════════════════════════════════════
            
            # 🔧 CORRECTIF: Utiliser perf_counter() partout pour cohérence temporelle
            t_rx_relative = time.perf_counter()  # Début du workflow (horloge relative cohérente)
            
            # ⏱️ Enregistrer début du workflow inter-étapes
            gateway.stats.mark_interstage_rx(frame_id, t_rx_relative)
            
            if use_gpu:
                try:
                    # ⏱️ Étape 1: CPU → GPU transfer
                    t1_start = time.perf_counter()
                    gpu_frame = prepare_frame_for_gpu(frame, device=gpu_device)
                    t1_end = time.perf_counter()
                    gateway.stats.mark_interstage_cpu_to_gpu(frame_id, t1_end)
                    cpu_to_gpu_ms = (t1_end - t1_start) * 1000.0
                    
                    # ⏱️ Étape 2: PROC (GPU processing)
                    t2_start = time.perf_counter()
                    tensor = gpu_frame.tensor
                    mask_tensor = (tensor > 0.5).float()  # Seuil à 0.5 (équivalent 128/255)
                    t2_end = time.perf_counter()
                    gateway.stats.mark_interstage_proc_done(frame_id, t2_end)
                    proc_gpu_ms = (t2_end - t2_start) * 1000.0
                    
                    # ⏱️ Étape 3: GPU → CPU transfer (final result)
                    t3_start = time.perf_counter()
                    mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    t3_end = time.perf_counter()
                    gateway.stats.mark_interstage_gpu_to_cpu(frame_id, t3_end)
                    gpu_to_cpu_ms = (t3_end - t3_start) * 1000.0
                    
                    # ✅ Calcul des latences inter-étapes par couples
                    rx_to_cpu_gpu = cpu_to_gpu_ms  # RX → CPU-to-GPU (t1_start était juste après RX)
                    cpu_gpu_to_proc = proc_gpu_ms   # CPU-to-GPU → PROC(GPU)
                    proc_to_gpu_cpu = gpu_to_cpu_ms # PROC(GPU) → GPU-to-CPU
                    # gpu_cpu_to_tx sera calculé automatiquement par mark_interstage_tx()
                    
                    # 📊 Log détaillé des métriques inter-étapes (toutes les 20 frames)
                    if frame_id % 20 == 0:
                        total_processing = cpu_to_gpu_ms + proc_gpu_ms + gpu_to_cpu_ms
                        LOG.info(f"[PROC-SIM]  Inter-stage latencies #{frame_id:03d}:")
                        LOG.info(f"  RX → CPU-to-GPU:    {rx_to_cpu_gpu:.2f}ms")
                        LOG.info(f"  CPU-to-GPU → PROC:  {cpu_gpu_to_proc:.2f}ms") 
                        LOG.info(f"  PROC → GPU-to-CPU:  {proc_to_gpu_cpu:.2f}ms")
                        LOG.info(f"  Total processing:   {total_processing:.2f}ms | {gpu_device}")
                        
                        # 📈 Afficher les statistiques cumulées si disponibles
                        try:
                            stats_snap = gateway.stats.snapshot()
                            interstage_samples = stats_snap.get('interstage_samples', 0)
                            if interstage_samples >= 5:  # Afficher seulement si assez d'échantillons
                                avg_proc = stats_snap.get('interstage_cpu_gpu_to_proc_ms', 0)
                                avg_total = (stats_snap.get('interstage_rx_to_cpu_gpu_ms', 0) + 
                                           avg_proc + 
                                           stats_snap.get('interstage_proc_to_gpu_cpu_ms', 0) + 
                                           stats_snap.get('interstage_gpu_cpu_to_tx_ms', 0))
                                LOG.info(f"  Moyennes cumulées ({interstage_samples} échantillons): PROC={avg_proc:.1f}ms, Total={avg_total:.1f}ms")
                        except Exception:
                            pass
                        
                except Exception as e:
                    LOG.warning(f"[PROC-SIM] GPU failed, fallback CPU: {e}")
                    # Fallback vers CPU (pas de métriques inter-étapes détaillées)
                    mask = (frame.image > 128).astype(np.uint8)
            else:
                # Traitement CPU classique (pas de transferts GPU)
                t_cpu_start = time.perf_counter()
                mask = (frame.image > 128).astype(np.uint8)
                t_cpu_end = time.perf_counter()
                cpu_proc_ms = (t_cpu_end - t_cpu_start) * 1000.0
                
                if frame_id % 20 == 0:
                    LOG.info(f"[PROC-SIM] CPU processing: {cpu_proc_ms:.2f}ms")
            
            # ✅ Timestamp final pour PROC→TX latency measurement
            t_proc_complete = time.perf_counter()
            meta = {
                "frame_id": frame_id,
                "ts": t_proc_complete,  # ⏱️ Timestamp de fin de PROC (début TX)
                "state": "VISIBLE",
            }
            
            # ⏱️ TX final - ceci appellera mark_tx() et mark_interstage_tx() automatiquement
            gateway.send_mask(mask, meta)
            
        except Exception as e:
            LOG.exception(f"[PROC-SIM] Error: {e}")
    LOG.info("[PROC-SIM] Thread stopped.")


# ──────────────────────────────────────────────
#  Lancement du Dashboard (optionnel)
# ──────────────────────────────────────────────
def start_dashboard_server(stop_event: threading.Event):
    """Lance le dashboard FastAPI dans un thread séparé - VERSION NON-BLOQUANTE."""
    try:
        import uvicorn
        import asyncio
        # 🎯 CORRECTIF: Utiliser le dashboard unifié corrigé au lieu de l'ancien
        from service.dashboard_unified import DashboardService, DashboardConfig
        
        LOG.info("🌐 Dashboard unifié démarré sur http://localhost:8050")
        
        # Configuration du dashboard unifié
        config = DashboardConfig(
            port=8050,
            host="0.0.0.0", 
            update_interval=1.0
        )
        
        dashboard_service = DashboardService(config)
        
        # 🎯 CORRECTIF: Lancer uvicorn de manière non-bloquante
        # Au lieu de dashboard_service.start() qui bloque, on utilise uvicorn.Server
        
        # Démarrer le thread de collecte de métriques
        dashboard_service.collector_thread.start()
        
        # Configuration uvicorn pour mode non-bloquant
        server_config = uvicorn.Config(
            app=dashboard_service.app,
            host=config.host,
            port=config.port,
            log_level="warning"  # Réduire le spam de logs
        )
        
        server = uvicorn.Server(server_config)
        
        # Lancer le serveur de manière non-bloquante
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Boucle jusqu'à stop_event
        loop.run_until_complete(server.serve())
            
    except ImportError:
        LOG.warning("⚠️ Dashboard non disponible (uvicorn/fastapi manquants)")
    except Exception as e:
        LOG.error(f"❌ Erreur dashboard: {e}")


# ──────────────────────────────────────────────
#  Test principal (RX → PROC → TX + Dashboard)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # NETTOYAGE : Supprimer les anciens logs
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NETTOYAGE DES LOGS PRECEDENTS")
    print("=" * 80)
    clean_old_logs()
    print()
    
    # ──────────────────────────────────────────────
    # LOGGING : Configurer le système de logging
    # ──────────────────────────────────────────────
    setup_logging()
    
    # ──────────────────────────────────────────────
    # GPU : Initialiser si disponible
    # ──────────────────────────────────────────────
    gpu_device = init_gpu_if_available()
    use_gpu = gpu_device != "cpu"
    
    # ──────────────────────────────────────────────
    # OPTIMISATION 1/3 : Activer timer Windows 1ms
    # ──────────────────────────────────────────────
    from utils.win_timer_resolution import enable_high_resolution_timer
    enable_high_resolution_timer()
    
    LOG.info("=" * 80)
    LOG.info("TEST PIPELINE AVEC DATASET REEL + GPU + DASHBOARD")
    LOG.info("=" * 80)
    LOG.info(f"Dataset: {DATASET_PATH}")
    LOG.info(f"GPU: {'[OK] Activé' if use_gpu else '[OFF] Désactivé'} (device={gpu_device})")
    LOG.info("Pipeline: RX (dataset) -> PROC (seuillage+GPU) -> TX (slicer_server)")
    LOG.info("Dashboard: http://localhost:8050 (lancement dans 2s...)")
    LOG.info("=" * 80)

    # Initialisation Gateway
    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    gateway._running = True  # mode offline

    # 🎯 Configurer la référence du gateway pour les métriques en temps réel
    from core.monitoring.monitor import set_active_gateway
    set_active_gateway(gateway)

    stop_event = threading.Event()
    frame_ready = threading.Event()

    # ──────────────────────────────────────────────
    # OPTIONNEL : Lancer le Dashboard
    # ──────────────────────────────────────────────
    ENABLE_DASHBOARD = True  # 🎯 Activé pour tester l'intégration gateway
    
    dashboard_thread = None
    if ENABLE_DASHBOARD:
        dashboard_thread = threading.Thread(
            target=start_dashboard_server,
            args=(stop_event,),
            daemon=True
        )
        dashboard_thread.start()
        time.sleep(2)  # Attendre que le dashboard démarre

    # Threads RX / PROC / TX
    LOG.info("Demarrage des threads...")
    
    # ✅ INSTRUMENTATION : Wrapper pour enregistrer TX timestamp (latency tracking)
    def tx_stats_callback(fps, bytes_count=0):
        """Callback TX qui enregistre les stats + timestamp TX pour calcul de latence."""
        gateway.update_tx_stats(fps, bytes_count)
        # Note: frame_id n'est pas disponible ici, mark_tx sera appelé ailleurs si nécessaire
    
    rx_thread = threading.Thread(
        target=read_dataset_images,
        args=(gateway, stop_event, frame_ready),
        kwargs={"loop_mode": True},  # 🔧 CORRECTIF: Boucle infinie pour dashboard persistant
        daemon=True,
        name="RX-Thread"
    )
    proc_thread = threading.Thread(
        target=simulate_processing,
        args=(gateway, stop_event, frame_ready, use_gpu, gpu_device),
        daemon=True,
        name="PROC-Thread"
    )
    tx_thread = threading.Thread(
        target=run_slicer_server,
        args=(
            gateway._outbox,
            stop_event,
            18945,
            tx_stats_callback,  # Callback pour update_tx_stats
            gateway.events.emit,
            gateway._tx_ready,
            gateway.stats  # ✅ INSTRUMENTATION : Passer stats pour mark_tx() (latency tracking)
        ),
        daemon=True,
        name="TX-Thread"
    )

    # Démarrage des threads
    # ✅ OPTIMISATION : Démarrage simultané (comme test_gateway_real_pipeline_mock.py)
    # Pas de sleep entre threads → évite backlog initial et latences artificielles
    LOG.info("Demarrage des threads...")
    proc_thread.start()  # PROC d'abord (prêt à recevoir)
    tx_thread.start()    # TX ensuite (prêt à envoyer)
    rx_thread.start()    # RX en dernier (commence l'injection)
    
    LOG.info("Tous les threads demarres !")
    LOG.info("=" * 80)

    # 🔧 Gestionnaire de signal pour arrêt propre
    def signal_handler(signum, frame):
        LOG.info(f"\n🛑 Signal {signum} reçu, arrêt en cours...")
        stop_event.set()
    
    # Enregistrer le gestionnaire de signal (Windows et Unix)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # 🔧 CORRECTIF: Boucle principale avec gestion Ctrl+C pour mode persistant
    try:
        LOG.info("🚀 Pipeline en cours... Ctrl+C pour arrêter")
        LOG.info("📊 Dashboard disponible sur: http://localhost:8050")
        
        # Boucle infinie jusqu'à Ctrl+C
        while True:
            time.sleep(1.0)  # Check toutes les secondes
            
            # Vérifier si les threads sont encore vivants
            if not rx_thread.is_alive():
                LOG.warning("RX thread terminated unexpectedly")
                break
            if not proc_thread.is_alive():
                LOG.warning("PROC thread terminated unexpectedly") 
                break
            if not tx_thread.is_alive():
                LOG.warning("TX thread terminated unexpectedly")
                break
                
    except KeyboardInterrupt:
        LOG.info("\n🛑 Ctrl+C détecté, arrêt en cours...")
    except Exception as e:
        LOG.error(f"❌ Erreur inattendue: {e}")

    # Arrêt propre
    LOG.info(">> Arrêt en cours...")
    stop_event.set()
    
    # Attendre les threads avec timeout
    LOG.info("Arrêt RX thread...")
    rx_thread.join(timeout=2.0)
    if rx_thread.is_alive():
        LOG.warning("RX thread ne s'est pas arrêté proprement")
        
    LOG.info("Arrêt PROC thread...")
    proc_thread.join(timeout=2.0)
    if proc_thread.is_alive():
        LOG.warning("PROC thread ne s'est pas arrêté proprement")
        
    LOG.info("Arrêt TX thread...")
    tx_thread.join(timeout=2.0)
    if tx_thread.is_alive():
        LOG.warning("TX thread ne s'est pas arrêté proprement")

    # Statistiques finales
    LOG.info("=" * 80)
    LOG.info("RESULTATS FINAUX - PIPELINE GPU-RÉSIDENT OPTIMISÉ")
    LOG.info("=" * 80)
    LOG.info(f"Outbox restante: {len(gateway._outbox)} items")
    
    # Afficher les stats du gateway si disponibles
    try:
        stats = gateway.stats.snapshot()
        
        # 📊 Statistiques générales
        LOG.info(f"\n STATISTIQUES GÉNÉRALES:")
        LOG.info(f"   RX FPS moyen: {stats.get('avg_fps_rx', 0):.1f}")
        LOG.info(f"   TX FPS moyen: {stats.get('avg_fps_tx', 0):.1f}")
        LOG.info(f"   Total bytes RX: {stats.get('bytes_rx', 0) / 1e6:.2f} MB")
        LOG.info(f"   Total bytes TX: {stats.get('bytes_tx', 0) / 1e6:.2f} MB")
        
        # 🎯 NOUVELLES MÉTRIQUES INTER-ÉTAPES DÉTAILLÉES
        interstage_samples = stats.get('interstage_samples', 0)
        if interstage_samples > 0:
            LOG.info(f"\n MÉTRIQUES INTER-ÉTAPES DÉTAILLÉES ({interstage_samples} échantillons):")
            LOG.info(f"   RX → CPU-to-GPU:    {stats.get('interstage_rx_to_cpu_gpu_ms', 0):.2f}ms (P95: {stats.get('interstage_rx_to_cpu_gpu_p95_ms', 0):.2f}ms)")
            LOG.info(f"   CPU-to-GPU → PROC:  {stats.get('interstage_cpu_gpu_to_proc_ms', 0):.2f}ms (P95: {stats.get('interstage_cpu_gpu_to_proc_p95_ms', 0):.2f}ms)")
            LOG.info(f"   PROC → GPU-to-CPU:  {stats.get('interstage_proc_to_gpu_cpu_ms', 0):.2f}ms (P95: {stats.get('interstage_proc_to_gpu_cpu_p95_ms', 0):.2f}ms)")
            LOG.info(f"   GPU-to-CPU → TX:    {stats.get('interstage_gpu_cpu_to_tx_ms', 0):.2f}ms (P95: {stats.get('interstage_gpu_cpu_to_tx_p95_ms', 0):.2f}ms)")
            
            # Calcul du total des étapes inter-médiaires
            total_interstage = (stats.get('interstage_rx_to_cpu_gpu_ms', 0) + 
                               stats.get('interstage_cpu_gpu_to_proc_ms', 0) + 
                               stats.get('interstage_proc_to_gpu_cpu_ms', 0) + 
                               stats.get('interstage_gpu_cpu_to_tx_ms', 0))
            LOG.info(f"    Total inter-étapes: {total_interstage:.2f}ms")
        else:
            LOG.info(f"\n AUCUNE MÉTRIQUE INTER-ÉTAPES (échantillons: {interstage_samples})")
            LOG.info(f"   Possible si mode CPU uniquement ou erreurs de traitement")
        
        # 🔍 Latences globales pour comparaison
        LOG.info(f"\n LATENCES GLOBALES:")
        LOG.info(f"   RX→TX moyenne: {stats.get('latency_ms_avg', 0):.2f}ms")
        LOG.info(f"   RX→TX P95: {stats.get('latency_ms_p95', 0):.2f}ms")
        LOG.info(f"   RX→TX max: {stats.get('latency_ms_max', 0):.2f}ms")
        LOG.info(f"   Échantillons latence: {stats.get('latency_samples', 0)}")
        
        # 🎯 Analyse de performance du pipeline GPU-résident
        if interstage_samples > 0:
            proc_ratio = stats.get('interstage_cpu_gpu_to_proc_ms', 0) / max(total_interstage, 0.001) * 100
            transfer_ratio = (stats.get('interstage_rx_to_cpu_gpu_ms', 0) + 
                             stats.get('interstage_proc_to_gpu_cpu_ms', 0)) / max(total_interstage, 0.001) * 100
            
            LOG.info(f"\n ANALYSE PIPELINE GPU-RÉSIDENT:")
            LOG.info(f"   Temps processing GPU: {proc_ratio:.1f}% du total")
            LOG.info(f"   Temps transferts GPU: {transfer_ratio:.1f}% du total")
            
            if proc_ratio > 60:
                LOG.info("   Pipeline optimisé: Processing domine les transferts")
                LOG.info("   Architecture GPU-résident validée avec succès")
            elif transfer_ratio > 40:
                LOG.info("   Optimisation possible: Transferts GPU élevés")
                LOG.info("   Recommandation: Vérifier les tailles de données et batching")
            else:
                LOG.info("   Pipeline équilibré")
        
        # 🏆 ÉVALUATION FINALE DU PIPELINE
        LOG.info(f"\n ÉVALUATION FINALE:")
        if use_gpu and interstage_samples > 0:
            if total_interstage < 15.0:
                LOG.info("   EXCELLENT: Pipeline GPU-résident très performant")
            elif total_interstage < 25.0:
                LOG.info("   BON: Pipeline GPU-résident performant")
            elif total_interstage < 40.0:
                LOG.info("   CORRECT: Pipeline fonctionnel, optimisations possibles")
            else:
                LOG.info("    LENT: Pipeline nécessite des optimisations")
            
            LOG.info(f"   Mode: GPU-résident optimisé (Phase 3)")
            LOG.info(f"   Temps total moyen: {total_interstage:.1f}ms")
        elif use_gpu:
            LOG.info("    Mode GPU activé mais pas de métriques inter-étapes")
            LOG.info("    Vérifier l'intégration des mark_interstage_*()")
        else:
            LOG.info("   Mode CPU classique (pas de GPU disponible)")
            LOG.info("   Pour de meilleures performances, utiliser un GPU compatible")
                
    except Exception as e:
        LOG.debug(f"Stats non disponibles: {e}")
    
    LOG.info("=" * 80)
    LOG.info(">>> Test terminé avec succès - Métriques inter-étapes collectées")
    LOG.info("=" * 80)
