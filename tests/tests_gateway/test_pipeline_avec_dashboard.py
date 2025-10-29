"""
Test Pipeline Intégré avec Dashboard
====================================

UN SEUL PROCESSUS qui fait tout :
1. Pipeline complète : RX (dataset) → PROC (GPU) → TX 
2. Dashboard web temps réel sur http://localhost:8050
3. Métriques inter-étapes détaillées GPU-résident

Usage :
    python test_pipeline_avec_dashboard.py
    
Puis ouvrir : http://localhost:8050
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
import numpy as np
from pathlib import Path
from PIL import Image
import glob
import asyncio

# ============================================================
# 🔧 UTF-8 SAFE MODE FOR WINDOWS CONSOLE
# ============================================================
import sys, io, os, locale

# Force UTF-8 code page for subprocesses
os.system("chcp 65001 >NUL")

# Force Python's stdout/stderr to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
import torch
from service.gateway.manager import IGTGateway
from service.slicer_server import run_slicer_server
from core.types import RawFrame, FrameMeta, Pose
from core.preprocessing.cpu_to_gpu import (
    init_transfer_runtime,
    prepare_frame_for_gpu,
)

# ──────────────────────────────────────────────
#  Logger asynchrone
# ──────────────────────────────────────────────
import logging.config, yaml
from core.monitoring import async_logging

LOG = logging.getLogger("igt.pipeline.test")

# ──────────────────────────────────────────────
#  Dashboard unifié (importé directement)
# ──────────────────────────────────────────────
from service.dashboard_unified import DashboardService, DashboardConfig

# ──────────────────────────────────────────────
#  DATASET PATH
# ──────────────────────────────────────────────
DATASET_PATH = Path(r"C:\Users\maxam\Desktop\TM\dataset\HUMERUS LATERAL XG SW_cropped\JPEGImages\Video_001")


def clean_old_logs():
    """Supprime tous les fichiers .log dans le dossier logs/"""
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        return
    
    deleted_count = 0
    for log_file in logs_dir.glob("*.log"):
        try:
            log_file.unlink()
            deleted_count += 1
        except Exception:
            pass
    
    if deleted_count > 0:
        print(f"[CLEAN] {deleted_count} fichier(s) log supprimé(s)")
    else:
        print("[CLEAN] Aucun log à supprimer")


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
    """Initialise le GPU si disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            LOG.info(f"[OK] CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            LOG.info("[WARNING] CUDA non disponible, utilisation CPU")
        
        init_transfer_runtime(device=device, pool_size=2, shape_hint=(512, 512))
        LOG.info(f"[OK] Runtime GPU initialisé (device={device})")
        return device
    except ImportError:
        LOG.warning("[WARNING] PyTorch non installé, utilisation CPU seulement")
        return "cpu"
    except Exception as e:
        LOG.warning(f"[WARNING] Erreur GPU, fallback CPU: {e}")
        return "cpu"


def read_dataset_images(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,
    fps: int = 30,  # Plus lent pour mieux visualiser
    target_size: tuple = (512, 512),
    loop_mode: bool = True  # Boucle infinie pour démo
):
    """Lit les images du dataset et les injecte dans la pipeline."""
    # Charger la liste des images
    if not DATASET_PATH.exists():
        LOG.error(f"Dataset path not found: {DATASET_PATH}")
        return
    
    image_files = sorted(glob.glob(str(DATASET_PATH / "*.jpg")))
    if not image_files:
        LOG.error(f"No JPEG images found in {DATASET_PATH}")
        return
    
    num_images = len(image_files)
    LOG.info(f"[DATASET-RX] Loaded {num_images} images, FPS: {fps} Hz")
    
    frame_id = 0
    image_idx = 0
    interval = 1.0 / fps
    next_frame_time = time.perf_counter()
    
    while not stop_event.is_set():
        img_path = image_files[image_idx]
        
        try:
            # Charger et redimensionner l'image
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("L")  # Grayscale
                pil_img = pil_img.resize(target_size, Image.BILINEAR)
                img = np.array(pil_img, dtype=np.uint8)
            
            # Créer la frame
            pose = Pose()
            ts = time.time()
            meta = FrameMeta(
                frame_id=frame_id,
                ts=ts,
                pose=pose,
                spacing=(0.3, 0.3, 1.0),
                orientation="UN",
                coord_frame="Image",
                device_name="Dataset",
            )
            frame = RawFrame(image=img, meta=meta)
            
            # Log périodique
            if frame_id % 30 == 0:
                LOG.info(f"[RX] Frame #{frame_id:03d} | {Path(img_path).name}")
            
            # Injection dans la pipeline
            gateway._inject_frame(frame)
            frame_ready.set()
            
            # Instrumentation RX
            gateway.stats.mark_rx(frame_id, ts)
            
            frame_id += 1
            image_idx += 1
            
            # Gestion boucle
            if image_idx >= num_images:
                if loop_mode:
                    image_idx = 0
                    LOG.info("[RX] Redémarrage boucle dataset")
                else:
                    LOG.info("[RX] Fin dataset, arrêt")
                    break
            
        except Exception as e:
            LOG.error(f"[RX] Erreur image {img_path}: {e}")
            image_idx = (image_idx + 1) % num_images
            continue
        
        # Sleep pour maintenir FPS
        next_frame_time += interval
        now = time.perf_counter()
        sleep_duration = next_frame_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
    
    LOG.info(f"[RX] Arrêt après {frame_id} frames")


def simulate_processing(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,
    use_gpu: bool = False,
    gpu_device: str = "cpu"
):
    """Traitement PROC avec seuillage et métriques GPU détaillées."""
    proc_type = "GPU thresholding" if use_gpu else "simple thresholding"
    LOG.info(f"[PROC] Thread started ({proc_type}, device={gpu_device})")
    
    while not stop_event.is_set():
        if not frame_ready.wait(timeout=0.01):
            continue
        frame_ready.clear()
        
        try:
            frame = gateway.receive_image()
            if frame is None:
                continue
            
            frame_id = frame.meta.frame_id
            t_rx = frame.meta.ts
            
            # ⏱️ Enregistrer début workflow inter-étapes
            gateway.stats.mark_interstage_rx(frame_id, t_rx)
            
            if use_gpu:
                try:
                    # Étape 1: CPU → GPU transfer
                    t1_start = time.perf_counter()
                    gpu_frame = prepare_frame_for_gpu(frame, device=gpu_device)
                    t1_end = time.perf_counter()
                    gateway.stats.mark_interstage_cpu_to_gpu(frame_id, t1_end)
                    
                    # Étape 2: PROC (GPU processing)
                    t2_start = time.perf_counter()
                    tensor = gpu_frame.tensor
                    mask_tensor = (tensor > 0.5).float()
                    t2_end = time.perf_counter()
                    gateway.stats.mark_interstage_proc_done(frame_id, t2_end)
                    
                    # Étape 3: GPU → CPU transfer
                    t3_start = time.perf_counter()
                    mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    t3_end = time.perf_counter()
                    gateway.stats.mark_interstage_gpu_to_cpu(frame_id, t3_end)
                    
                    # Log périodique des métriques
                    if frame_id % 50 == 0:
                        cpu_to_gpu_ms = (t1_end - t1_start) * 1000.0
                        proc_gpu_ms = (t2_end - t2_start) * 1000.0
                        gpu_to_cpu_ms = (t3_end - t3_start) * 1000.0
                        total_ms = cpu_to_gpu_ms + proc_gpu_ms + gpu_to_cpu_ms
                        
                        LOG.info(f"[PROC] Frame #{frame_id}: GPU={gpu_device}")
                        LOG.info(f"  CPU→GPU: {cpu_to_gpu_ms:.2f}ms")
                        LOG.info(f"  PROC:    {proc_gpu_ms:.2f}ms") 
                        LOG.info(f"  GPU→CPU: {gpu_to_cpu_ms:.2f}ms")
                        LOG.info(f"  Total:   {total_ms:.2f}ms")
                        
                except Exception as e:
                    LOG.warning(f"[PROC] GPU failed, fallback CPU: {e}")
                    mask = (frame.image > 128).astype(np.uint8)
            else:
                # Traitement CPU
                mask = (frame.image > 128).astype(np.uint8)
            
            # Préparer metadata TX
            t_proc_complete = time.perf_counter()
            meta = {
                "frame_id": frame_id,
                "ts": t_proc_complete,
                "state": "VISIBLE",
            }
            
            # Envoyer vers TX
            gateway.send_mask(mask, meta)
            
        except Exception as e:
            LOG.exception(f"[PROC] Error: {e}")
    
    LOG.info("[PROC] Thread stopped")


class PipelineWithDashboard:
    """Pipeline intégrée avec dashboard dans le même processus."""
    
    def __init__(self):
        self.gateway = None
        self.dashboard_service = None
        self.stop_event = threading.Event()
        self.frame_ready = threading.Event()
        
        # Threads
        self.rx_thread = None
        self.proc_thread = None
        self.tx_thread = None
        
        # Config
        self.use_gpu = False
        self.gpu_device = "cpu"
    
    def setup(self):
        """Initialisation complète."""
        print("=" * 80)
        print("PIPELINE INTÉGRÉE AVEC DASHBOARD")
        print("=" * 80)
        
        # Nettoyage logs
        clean_old_logs()
        
        # Logging
        setup_logging()
        
        # GPU
        self.gpu_device = init_gpu_if_available()
        self.use_gpu = self.gpu_device != "cpu"
        
        # Timer Windows
        from utils.win_timer_resolution import enable_high_resolution_timer
        enable_high_resolution_timer()
        
        # Gateway
        self.gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=30.0)
        self.gateway._running = True  # mode offline
        
        # 🎯 CRUCIAL: Enregistrer le gateway pour les métriques temps réel
        from core.monitoring.monitor import set_active_gateway
        set_active_gateway(self.gateway)
        
        LOG.info(f"GPU: {'[OK] Activé' if self.use_gpu else '[OFF] Désactivé'} (device={self.gpu_device})")
        LOG.info("Pipeline: RX (dataset) → PROC (seuillage+GPU) → TX (slicer)")
        LOG.info("Dashboard: http://localhost:8050")
        
        return True
    
    def start_dashboard(self):
        """Démarre le dashboard dans un thread séparé."""
        try:
            config = DashboardConfig(
                port=8050,
                host="0.0.0.0",
                update_interval=1.0
            )
            
            self.dashboard_service = DashboardService(config)
            
            # Démarrer le collecteur de métriques
            self.dashboard_service.collector_thread.start()
            
            # Démarrer le serveur web dans un thread
            def run_dashboard():
                import uvicorn
                server_config = uvicorn.Config(
                    app=self.dashboard_service.app,
                    host=config.host,
                    port=config.port,
                    log_level="warning"
                )
                server = uvicorn.Server(server_config)
                
                # Créer event loop pour ce thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    loop.run_until_complete(server.serve())
                except Exception as e:
                    LOG.error(f"Erreur dashboard: {e}")
                finally:
                    loop.close()
            
            dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            dashboard_thread.start()
            
            LOG.info("🌐 Dashboard démarré sur http://localhost:8050")
            time.sleep(2)  # Attendre que le serveur démarre
            
            return True
            
        except Exception as e:
            LOG.error(f"Erreur démarrage dashboard: {e}")
            return False
    
    def start_pipeline(self):
        """Démarre les threads de la pipeline."""
        
        # Callback TX pour stats
        def tx_stats_callback(fps, bytes_count=0):
            self.gateway.update_tx_stats(fps, bytes_count)
        
        # Créer les threads
        self.rx_thread = threading.Thread(
            target=read_dataset_images,
            args=(self.gateway, self.stop_event, self.frame_ready),
            kwargs={"fps": 30, "loop_mode": True},  # 30 FPS en boucle
            daemon=True,
            name="RX-Thread"
        )
        
        self.proc_thread = threading.Thread(
            target=simulate_processing,
            args=(self.gateway, self.stop_event, self.frame_ready, self.use_gpu, self.gpu_device),
            daemon=True,
            name="PROC-Thread"
        )
        
        self.tx_thread = threading.Thread(
            target=run_slicer_server,
            args=(
                self.gateway._outbox,
                self.stop_event,
                18945,
                tx_stats_callback,
                self.gateway.events.emit,
                self.gateway._tx_ready,
                self.gateway.stats
            ),
            daemon=True,
            name="TX-Thread"
        )
        
        # Démarrer dans l'ordre optimal
        LOG.info("Démarrage pipeline...")
        self.proc_thread.start()
        self.tx_thread.start() 
        self.rx_thread.start()
        
        LOG.info("✅ Pipeline démarrée !")
        return True
    
    def run(self):
        """Méthode principale - tout en un."""
        try:
            # 1. Setup
            if not self.setup():
                return False
            
            # 2. Dashboard
            if not self.start_dashboard():
                LOG.warning("Dashboard non démarré, continue sans")
            
            # 3. Pipeline
            if not self.start_pipeline():
                return False
            
            # 4. Monitoring
            LOG.info("=" * 80)
            LOG.info("🚀 SYSTÈME DÉMARRÉ !")
            LOG.info("📊 Dashboard: http://localhost:8050")
            LOG.info("⚡ Pipeline: RX → PROC → TX en cours...")
            LOG.info("🛑 Ctrl+C pour arrêter")
            LOG.info("=" * 80)
            
            # Affichage stats périodique
            last_stats_time = time.time()
            
            while True:
                time.sleep(5)  # Check toutes les 5s
                
                # Stats périodiques
                now = time.time()
                if now - last_stats_time > 30:  # Toutes les 30s
                    try:
                        stats = self.gateway.stats.snapshot()
                        interstage_samples = stats.get('interstage_samples', 0)
                        
                        LOG.info(f"📊 Stats: {interstage_samples} échantillons inter-étapes")
                        if interstage_samples > 0:
                            LOG.info(f"  RX→GPU: {stats.get('interstage_rx_to_cpu_gpu_ms', 0):.1f}ms")
                            LOG.info(f"  PROC:   {stats.get('interstage_cpu_gpu_to_proc_ms', 0):.1f}ms") 
                            LOG.info(f"  GPU→TX: {stats.get('interstage_gpu_cpu_to_tx_ms', 0):.1f}ms")
                        
                        last_stats_time = now
                    except Exception:
                        pass
                
        except KeyboardInterrupt:
            LOG.info("\n🛑 Arrêt demandé par l'utilisateur")
        except Exception as e:
            LOG.error(f"❌ Erreur système: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Arrêt propre de tous les composants."""
        LOG.info("🛑 Arrêt en cours...")
        
        # Signal d'arrêt
        self.stop_event.set()
        
        # Attendre les threads
        if self.rx_thread:
            self.rx_thread.join(timeout=2.0)
        if self.proc_thread:
            self.proc_thread.join(timeout=2.0) 
        if self.tx_thread:
            self.tx_thread.join(timeout=2.0)
        
        # Dashboard
        if self.dashboard_service:
            try:
                self.dashboard_service.stop()
            except Exception:
                pass
        
        LOG.info("✅ Arrêt terminé")


if __name__ == "__main__":
    pipeline = PipelineWithDashboard()
    pipeline.run()