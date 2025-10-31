
"""
proc_simulator.py
-----------------
Simulation du traitement pipeline (CPU ou GPU).
"""
import time
import logging
import threading
import numpy as np

from core.preprocessing.cpu_to_gpu import prepare_frame_for_gpu
from service.gateway.manager import IGTGateway  #

LOG = logging.getLogger("igt.mock.proc")

# ──────────────────────────────────────────────
#  Traitement PROC (seuillage) - INCHANGÉ
# ──────────────────────────────────────────────
def simulate_processing(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,
    use_gpu: bool = True,
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
                    
                    # 📊 Log détaillé des métriques inter-étapes (toutes les 10 frames)
                    if frame_id % 10 == 0:
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
                
                if frame_id % 10 == 0:
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