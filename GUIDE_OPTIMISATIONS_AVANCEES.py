"""
GUIDE COMPLET DES OPTIMISATIONS AVANCÉES POUR RÉDUIRE LES PICS
================================================================

Ce document explique les 6 catégories d'optimisations proposées,
où les appliquer, et leur pertinence pour votre pipeline.

Actuellement vos performances :
- RX→PROC : 0.10ms moyenne (93% à 0ms) - EXCELLENT ✅
- PROC→TX : 0.42ms moyenne (67% à 0ms) - EXCELLENT ✅
- Pics observés : 1-3ms (7-8% des frames)
"""

# ============================================================================
# 1️⃣ TIMER WINDOWS (Résolution 1ms au lieu de 15.6ms)
# ============================================================================

"""
❓ QU'EST-CE QUE ÇA FAIT ?
--------------------------
Windows par défaut a un timer system de 15.6ms (64 Hz).
Quand vous faites time.sleep(0.01), Windows peut réellement dormir 15.6ms !

timeBeginPeriod(1) demande à Windows d'augmenter la résolution à 1ms.
→ time.sleep(0.01) dormira vraiment ~10-11ms au lieu de 15.6ms.

📍 OÙ L'APPLIQUER ?
------------------
Au TOUT DÉBUT de votre main(), avant de lancer les threads.

🔧 COMMENT ?
-----------
"""

# Créer fichier : src/utils/win_timer_resolution.py
import atexit
import ctypes
import sys

def enable_high_resolution_timer():
    """Active timer Windows 1ms (au lieu de 15.6ms par défaut)"""
    if sys.platform != "win32":
        return  # Seulement Windows
    
    try:
        winmm = ctypes.WinDLL("winmm")
        result = winmm.timeBeginPeriod(1)  # 1ms resolution
        if result == 0:
            print("✅ Timer Windows configuré à 1ms")
            # Restaurer à la fin du programme
            atexit.register(lambda: winmm.timeEndPeriod(1))
        else:
            print(f"⚠️  timeBeginPeriod échoué (code {result})")
    except Exception as e:
        print(f"⚠️  Impossible de configurer timer: {e}")

"""
💡 UTILISATION dans votre test :
"""

# Dans test_gateway_real_pipeline_mock.py, ligne 1
from utils.win_timer_resolution import enable_high_resolution_timer

if __name__ == "__main__":
    enable_high_resolution_timer()  # ← AJOUTER EN PREMIER
    
    LOG.info("Initialisation du IGTGateway...")
    # ... reste du code ...

"""
📊 IMPACT ATTENDU :
------------------
- Votre diagnostic montrait : intervalles RX de 12.1ms au lieu de 10ms
- Avec timer 1ms : intervalles RX de 10-11ms (beaucoup plus régulier)
- Réduction des pics liés au scheduler Windows

⚖️ COÛT/BÉNÉFICE :
-----------------
✅ RECOMMANDÉ
- Complexité : TRÈS FAIBLE (5 lignes de code)
- Gain : MOYEN (réduit jitter Windows)
- Risque : AUCUN (restauré automatiquement)
- Side-effect : Augmente légèrement consommation CPU système

🎯 VERDICT : À FAIRE ! Simple et efficace.
"""


# ============================================================================
# 2️⃣ BOUCLE À HORLOGE COMPENSÉE (sleep précis)
# ============================================================================

"""
❓ QU'EST-CE QUE ÇA FAIT ?
--------------------------
Au lieu de :
    while True:
        do_work()
        time.sleep(0.01)  # ← Dérive s'accumule

On fait :
    next_time = now + 0.01
    while True:
        do_work()
        sleep_until(next_time)  # ← Compense automatiquement
        next_time += 0.01

→ Élimine la dérive temporelle (drift)

📍 OÙ L'APPLIQUER ?
------------------
Dans simulate_frame_source() (RX thread) pour générer à 100Hz précis

🔧 COMMENT ?
-----------
"""

# Modifier test_gateway_real_pipeline_mock.py, fonction simulate_frame_source()

def simulate_frame_source(
    gateway,
    stop_event,
    frame_ready,
    fps: int = 100
):
    """Génère des RawFrame avec horloge compensée (pas de dérive)"""
    import time
    
    frame_id = 0
    interval = 1.0 / fps  # 0.01 pour 100 Hz
    
    # 🔬 OPTIMISATION : Horloge compensée
    next_time = time.perf_counter() + interval
    
    LOG.info(f"[RX-SIM] Frame generator started at {fps} Hz (compensated clock)")

    while not stop_event.is_set():
        # Générer frame...
        img = (np.random.rand(512, 512) * 255).astype(np.uint8)
        # ... (reste du code identique)
        
        LOG.info(f"[RX-SIM] Generated frame #{frame_id:03d}")
        gateway._inject_frame(frame)
        frame_ready.set()
        frame_id += 1
        
        # 🔬 SLEEP COMPENSÉ (au lieu de time.sleep(interval))
        now = time.perf_counter()
        sleep_duration = next_time - now
        
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        
        # Rattrapage si on est en retard
        while next_time < time.perf_counter():
            next_time += interval
        
        next_time += interval

"""
📊 IMPACT ATTENDU :
------------------
AVANT : intervalles RX de 10-14ms (moyenne 12.1ms) - irrégulier
APRÈS : intervalles RX de 9.9-10.1ms (moyenne 10.0ms) - très régulier

⚖️ COÛT/BÉNÉFICE :
-----------------
✅ RECOMMANDÉ
- Complexité : FAIBLE (10 lignes modifiées)
- Gain : MOYEN (horloge stable = moins de jitter)
- Risque : AUCUN

🎯 VERDICT : À FAIRE ! Meilleure régularité.
"""


# ============================================================================
# 3️⃣ PRIORITÉ PROCESS/THREADS (Windows scheduling)
# ============================================================================

"""
❓ QU'EST-CE QUE ÇA FAIT ?
--------------------------
Demande à Windows de donner PLUS de temps CPU à votre process/threads.

HIGH_PRIORITY_CLASS : votre process passe avant les process normaux
THREAD_PRIORITY_ABOVE_NORMAL : RX/PROC passent avant les threads normaux

→ Réduit les préemptions par d'autres applications

📍 OÙ L'APPLIQUER ?
------------------
- Process priority : Au début du main()
- Thread priority : Au début de chaque fonction thread (RX, PROC, TX)

🔧 COMMENT ?
-----------
"""

# Option A : Priorité PROCESS (simple, global)
import psutil

if __name__ == "__main__":
    # Hausse priorité du process entier
    p = psutil.Process()
    p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows HIGH_PRIORITY
    LOG.info(f"✅ Process priority: {p.nice()}")
    
    # Optionnel : réserver des cœurs CPU
    # p.cpu_affinity([2, 3, 4, 5])  # Laisse 0-1 au système

# Option B : Priorité par THREAD (plus fin)
import ctypes

def boost_thread_priority():
    """Augmente priorité du thread actuel"""
    try:
        THREAD_PRIORITY_ABOVE_NORMAL = 1
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentThread()
        result = kernel32.SetThreadPriority(handle, THREAD_PRIORITY_ABOVE_NORMAL)
        if result:
            print("✅ Thread priority boosted")
        return result
    except Exception as e:
        print(f"⚠️  Failed to boost thread: {e}")
        return False

# Dans simulate_frame_source() et simulate_processing()
def simulate_frame_source(...):
    boost_thread_priority()  # ← AJOUTER EN PREMIER
    LOG.info("[RX-SIM] Frame generator started...")
    # ... reste du code

def simulate_processing(...):
    boost_thread_priority()  # ← AJOUTER EN PREMIER
    LOG.info("[PROC-SIM] Thread started...")
    # ... reste du code

"""
📊 IMPACT ATTENDU :
------------------
- Moins de préemptions par Chrome, Antivirus, etc.
- Pics liés au scheduler réduits de 10-15ms à 5-10ms

⚖️ COÛT/BÉNÉFICE :
-----------------
⚠️ À ÉVALUER
- Complexité : FAIBLE
- Gain : MOYEN (mais peut affecter le reste du système)
- Risque : MOYEN (peut ralentir autres applications)

🎯 VERDICT : TESTER d'abord, mais PAS en production sans validation.
     Préférer ABOVE_NORMAL, éviter REALTIME (dangereux).
"""


# ============================================================================
# 4️⃣ ÉVITER OVERSUBSCRIPTION CPU (Torch/NumPy threads)
# ============================================================================

"""
❓ QU'EST-CE QUE ÇA FAIT ?
--------------------------
NumPy et PyTorch utilisent par défaut TOUS les cœurs CPU pour les calculs.
→ Créent plein de threads internes qui se battent avec RX/PROC/TX
→ Pics périodiques quand BLAS/MKL s'exécute

OMP_NUM_THREADS=1 limite NumPy/Torch à 1 thread par opération.

📍 OÙ L'APPLIQUER ?
------------------
AVANT d'importer numpy/torch (donc tout en haut du fichier)

🔧 COMMENT ?
-----------
"""

# Dans test_gateway_real_pipeline_mock.py, AVANT les imports numpy/torch
import os

# 🔬 OPTIMISATION : Limite threads NumPy/Torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Si vous utilisez PyTorch (pas dans votre test actuel)
# import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

import numpy as np  # ← Maintenant NumPy utilisera 1 thread

"""
📊 IMPACT ATTENDU :
------------------
- Réduit contention CPU (moins de threads qui se battent)
- Peut RALENTIR les calculs NumPy complexes (matrix multiply, etc.)
- Votre cas : np.random.rand() → impact négligeable

⚖️ COÛT/BÉNÉFICE :
-----------------
✅ RECOMMANDÉ pour votre cas
- Complexité : TRÈS FAIBLE (3 lignes)
- Gain : FAIBLE à MOYEN (dépend de vos calculs NumPy)
- Risque : FAIBLE (peut ralentir gros calculs matriciels)

🎯 VERDICT : À FAIRE si vous n'avez pas de calculs NumPy intensifs.
"""


# ============================================================================
# 5️⃣ LOGGING BUFFERISÉ (MemoryHandler)
# ============================================================================

"""
❓ QU'EST-CE QUE ÇA FAIT ?
--------------------------
Actuellement : QueueHandler → QueueListener → flush disque IMMÉDIAT
                              ↑ Bloque le GIL pendant write()

MemoryHandler : accumule 10000 logs en RAM → flush RARE
→ Réduit les pauses I/O de 0.5-2ms à quasi zéro

📍 OÙ L'APPLIQUER ?
------------------
Dans src/config/logging.yaml OU dans setup_async_logging()

🔧 COMMENT ?
-----------
"""

# Option : Modifier src/core/monitoring/async_logging.py

import logging
from logging.handlers import RotatingFileHandler, MemoryHandler

def setup_async_logging(yaml_cfg=None):
    # ... code existant ...
    
    # Au lieu de ajouter directement RotatingFileHandler
    file_handler = RotatingFileHandler(
        "logs/pipeline.log",
        maxBytes=5_000_000,
        backupCount=3,
        delay=True  # ← N'ouvre pas le fichier immédiatement
    )
    file_handler.setFormatter(formatter)
    
    # 🔬 OPTIMISATION : Buffer en mémoire
    memory_handler = MemoryHandler(
        capacity=10000,              # Accumule 10k logs avant flush
        flushLevel=logging.ERROR,    # Flush immédiat si ERROR+
        target=file_handler
    )
    
    root_logger.addHandler(memory_handler)  # Au lieu de file_handler

"""
📊 IMPACT ATTENDU :
------------------
- Réduit pics liés aux flush disque de ~2ms à ~0ms
- Logs peuvent être perdus si crash (max 10k messages)

⚖️ COÛT/BÉNÉFICE :
-----------------
⚠️ À ÉVALUER
- Complexité : MOYENNE (modifier architecture logging)
- Gain : MOYEN (réduit pics I/O)
- Risque : MOYEN (perte logs en cas de crash)

🎯 VERDICT : OPTIONNEL. Vos logs sont déjà async avec QueueHandler.
     Gain marginal vs risque de perte de logs.
"""


# ============================================================================
# 6️⃣ GARBAGE COLLECTOR (GC manuel)
# ============================================================================

"""
❓ QU'EST-CE QUE ÇA FAIT ?
--------------------------
Python GC s'exécute automatiquement et bloque TOUS les threads (1-5ms).

gc.disable() : désactive GC automatique
gc.collect() : déclenche GC manuellement dans un thread supervisor

→ Contrôle QUAND le GC s'exécute (pendant les phases idle)

📍 OÙ L'APPLIQUER ?
------------------
Au début du main() + créer un thread supervisor

🔧 COMMENT ?
-----------
"""

import gc
import threading

if __name__ == "__main__":
    # Désactiver GC automatique
    gc.disable()
    LOG.info("✅ GC automatique désactivé")
    
    # Créer thread supervisor GC
    def gc_supervisor():
        while not stop_event.is_set():
            stop_event.wait(timeout=2.0)  # Toutes les 2 secondes
            collected = gc.collect()
            if collected > 0:
                LOG.debug(f"[GC] Collected {collected} objects")
    
    gc_thread = threading.Thread(target=gc_supervisor, daemon=True, name="GC-Supervisor")
    gc_thread.start()
    
    # ... lancer RX/PROC/TX ...

"""
📊 IMPACT ATTENDU :
------------------
- Élimine les pauses GC aléatoires pendant le traitement frames
- GC s'exécute toutes les 2s pendant les "creux" de charge

⚖️ COÛT/BÉNÉFICE :
-----------------
⚠️ AVANCÉ
- Complexité : MOYENNE
- Gain : MOYEN (élimine pics GC aléatoires)
- Risque : MOYEN (consommation RAM peut augmenter)

🎯 VERDICT : OPTIONNEL. Tester si vous voyez des pics à 5ms.
     Pour l'instant vos pics sont 1-3ms → pas urgent.
"""


# ============================================================================
# 📊 RÉCAPITULATIF : QUOI FAIRE DANS VOTRE CAS ?
# ============================================================================

"""
VOS PERFORMANCES ACTUELLES :
- RX→PROC : 0.10ms moyenne (EXCELLENT ✅)
- PROC→TX : 0.42ms moyenne (EXCELLENT ✅)
- Pics : 1-3ms (7-8% des frames)

OPTIMISATIONS RECOMMANDÉES (par ordre de priorité) :
====================================================

🥇 PRIORITÉ 1 (À FAIRE - gain certain, risque faible)
-----------------------------------------------------
✅ 1. Timer Windows 1ms (win_timer_resolution.py)
   → Réduit jitter Windows de 15.6ms à 1ms
   → Complexité : 5 min
   → Gain attendu : intervalles RX de 12.1ms → 10.0ms

✅ 2. Horloge compensée dans RX thread
   → Élimine dérive temporelle
   → Complexité : 15 min
   → Gain attendu : intervalles très réguliers (9.9-10.1ms)

✅ 3. Limite threads NumPy (OMP_NUM_THREADS=1)
   → Réduit contention CPU
   → Complexité : 2 min
   → Gain attendu : moins de pics liés aux calculs NumPy

IMPACT COMBINÉ : Réduction des pics de ~30-40%
               (1-3ms → 0.5-2ms pour 90% des frames)


🥈 PRIORITÉ 2 (OPTIONNEL - à tester si besoin)
----------------------------------------------
⚠️ 4. Priorité process (psutil HIGH_PRIORITY_CLASS)
   → Réduit préemptions par autres apps
   → Complexité : 5 min
   → Risque : Peut ralentir le reste du système
   → À TESTER en environnement de dev uniquement

⚠️ 5. GC manuel (gc.disable() + supervisor)
   → Élimine pics GC aléatoires
   → Complexité : 20 min
   → Risque : Augmente RAM
   → Utile si vous voyez des pics à 5ms (pas votre cas)


🥉 PRIORITÉ 3 (PAS RECOMMANDÉ pour l'instant)
---------------------------------------------
❌ 6. MemoryHandler pour logging
   → Gain marginal (logs déjà async)
   → Risque de perte logs en cas de crash

❌ 7. Priorité threads individuels
   → Complexité plus élevée que priorité process
   → Gain similaire


PLAN D'ACTION SUGGÉRÉ :
======================
1. Implémenter PRIORITÉ 1 (1h de travail total)
2. Tester et mesurer l'impact
3. Si pics encore > 2ms : tester PRIORITÉ 2
4. Sinon : STOP, performances excellentes !
"""

if __name__ == "__main__":
    print(__doc__)
