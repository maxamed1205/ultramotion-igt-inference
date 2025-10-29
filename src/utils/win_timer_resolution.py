"""
High-Resolution Timer for Windows
==================================

Windows par défaut utilise un timer système de 15.6ms (64 Hz).
Ce module configure le timer à 1ms pour des sleep() plus précis.

Usage:
    from utils.win_timer_resolution import enable_high_resolution_timer
    
    if __name__ == "__main__":
        enable_high_resolution_timer()  # Appeler au début du main()
        # ... reste du code ...

Impact:
    - time.sleep(0.01) dormira ~10-11ms au lieu de ~15-16ms
    - Réduit le jitter du scheduler Windows
    - Améliore la régularité des intervalles entre frames

Note: Le timer est automatiquement restauré à la fin du programme.
"""

import atexit
import ctypes
import sys
import logging

LOG = logging.getLogger(__name__)


def enable_high_resolution_timer() -> bool:
    """
    Active le timer Windows haute résolution (1ms au lieu de 15.6ms).
    
    Returns:
        bool: True si succès, False sinon (ou si non-Windows)
    
    Note:
        - Fonctionne uniquement sur Windows
        - Le timer est automatiquement restauré via atexit
        - Augmente légèrement la consommation CPU système (~1-2%)
    """
    if sys.platform != "win32":
        LOG.debug("High-resolution timer: non-Windows platform, skipped")
        return False
    
    try:
        # Charger winmm.dll (Windows Multimedia API)
        winmm = ctypes.WinDLL("winmm")
        
        # Demander résolution 1ms (minimum supporté par Windows)
        # timeBeginPeriod retourne 0 si succès, autre valeur si erreur
        result = winmm.timeBeginPeriod(1)
        
        if result == 0:
            LOG.info("✅ Timer Windows configuré à 1ms (au lieu de 15.6ms)")
            
            # Enregistrer la restauration automatique à la fin du programme
            atexit.register(lambda: winmm.timeEndPeriod(1))
            LOG.debug("Timer cleanup registered via atexit")
            
            return True
        else:
            LOG.warning(f"⚠️  timeBeginPeriod(1) failed with code {result}")
            return False
            
    except OSError as e:
        LOG.error(f"❌ Failed to load winmm.dll: {e}")
        return False
    except Exception as e:
        LOG.error(f"❌ Unexpected error setting timer resolution: {e}")
        return False


def get_timer_resolution() -> float:
    """
    Récupère la résolution actuelle du timer Windows (en ms).
    
    Returns:
        float: Résolution en millisecondes, ou -1.0 si erreur
    
    Note:
        Fonctionne uniquement sur Windows. Sur autres OS, retourne -1.0.
    """
    if sys.platform != "win32":
        return -1.0
    
    try:
        winmm = ctypes.WinDLL("winmm")
        
        # Structure TIMECAPS
        class TIMECAPS(ctypes.Structure):
            _fields_ = [
                ("wPeriodMin", ctypes.c_uint),
                ("wPeriodMax", ctypes.c_uint),
            ]
        
        caps = TIMECAPS()
        result = winmm.timeGetDevCaps(ctypes.byref(caps), ctypes.sizeof(caps))
        
        if result == 0:  # TIMERR_NOERROR
            return float(caps.wPeriodMin)
        else:
            return -1.0
            
    except Exception:
        return -1.0


if __name__ == "__main__":
    # Test du module
    print("🔬 Test du module win_timer_resolution")
    print("=" * 50)
    
    # Afficher résolution avant
    res_before = get_timer_resolution()
    if res_before > 0:
        print(f"Résolution timer AVANT : {res_before} ms")
    else:
        print("Impossible de lire la résolution timer")
    
    # Activer haute résolution
    success = enable_high_resolution_timer()
    
    if success:
        print("✅ Timer haute résolution activé")
        
        # Test de précision avec time.sleep()
        import time
        print("\nTest de précision sleep() :")
        
        for i in range(5):
            start = time.perf_counter()
            time.sleep(0.01)  # 10ms visé
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  sleep(0.01) → {elapsed:.2f} ms")
    else:
        print("❌ Échec activation timer haute résolution")
