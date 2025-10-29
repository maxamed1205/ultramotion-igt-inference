"""Thread-safe stats container for the gateway.

This module extends the existing GatewayStats with:
 - a frozen dataclass snapshot (GatewayStatsSnapshot) convertible to dict,
 - rolling FPS windows configurable at construction time (deque maxlen),
 - RX->TX latency measurement via mark_rx/mark_tx with a bounded reservoir,
 - reset() to clear volatile counters,
 - global started_at and uptime values.

The public API from previous versions is preserved (update_rx, update_tx,
snapshot, set_last_rx_ts). New options are optional constructor args, so existing
callers remain compatible.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Deque, Dict, Any, List, Optional, TypedDict

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

LOG = logging.getLogger("igt.gateway.stats")

# --- Dictionnaire typé (TypedDict) représentant la structure complète du snapshot ---
class GatewayStatsDict(TypedDict):
    """Typed dictionary représentant la structure complète d’un snapshot GatewayStats.

    Ce type sert à documenter et typer la sortie des méthodes `to_dict()` et `snapshot()`.
    Chaque clé correspond à une métrique précise, dont la signification est détaillée ci-dessous.
    """

    # --- Débits instantanés ---
    fps_rx: float          # fréquence instantanée de réception (frames/s) – mesurée sur la dernière frame reçue
    fps_tx: float          # fréquence instantanée d’envoi (frames/s) – cadence actuelle de transmission vers 3D Slicer

    # --- Moyennes glissantes ---
    avg_fps_rx: float      # débit moyen d’entrée – moyenne glissante des N dernières mesures (fenêtre _rolling_fps_rx)
    avg_fps_tx: float      # débit moyen de sortie – moyenne glissante des N dernières mesures (fenêtre _rolling_fps_tx)

    # --- Compteurs d’octets ---
    bytes_rx: int          # taille en octets du dernier message reçu (image ou tracking IGTLink)
    bytes_tx: int          # taille en octets du dernier message envoyé (masque + métadonnées)
    total_bytes_rx: int    # cumul total d’octets reçus depuis le démarrage du service (toutes frames confondues)
    total_bytes_tx: int    # cumul total d’octets envoyés depuis le démarrage du service (tous masques confondus)

    # --- Horodatages d’activité ---
    last_rx_ts: float      # timestamp (epoch) de la dernière frame reçue depuis PlusServer
    last_update_rx: float  # horodatage système (time.time) du dernier appel à update_rx() – utile pour détecter une inactivité RX
    last_update_tx: float  # horodatage système (time.time) du dernier appel à update_tx() – utile pour détecter une inactivité TX

    # --- Données temporelles globales ---
    started_at: float      # timestamp du démarrage du module GatewayStats – point de référence pour l’uptime
    uptime_s: float        # durée écoulée depuis started_at (temps total de fonctionnement en secondes)

    # --- Statistiques de latence RX→TX ---
    latency_ms_avg: float  # latence moyenne RX→TX (en millisecondes) calculée sur les derniers échantillons
    latency_ms_p95: float  # latence au 95ᵉ percentile – 95 % des frames ont une latence inférieure à cette valeur
    latency_ms_max: float  # latence maximale observée (plus grand délai RX→TX enregistré)
    latency_samples: int   # nombre total d’échantillons de latence conservés dans le buffer (max ≈ 200)

    # --- Indicateurs d'anomalie et de suivi ---
    latency_orphans: int   # nombre de frames TX sans RX correspondant (frames "orphelines" dues à une désynchro ou un drop)
    snapshot_count: int    # compteur global de snapshots générés depuis le démarrage (utile pour suivre la fréquence de reporting)
    
    # 🎯 ═══════════════════════════════════════════════════════════════════
    # NOUVELLES MÉTRIQUES INTER-ÉTAPES (Workflow GPU-résident détaillé)
    # ═══════════════════════════════════════════════════════════════════
    
    # --- Latences inter-étapes moyennes (en ms) ---
    interstage_rx_to_cpu_gpu_ms: float      # RX → CPU-to-GPU transfer latency
    interstage_cpu_gpu_to_proc_ms: float    # CPU-to-GPU → PROC(GPU) latency
    interstage_proc_to_gpu_cpu_ms: float    # PROC(GPU) → GPU-to-CPU transfer latency
    interstage_gpu_cpu_to_tx_ms: float      # GPU-to-CPU → TX latency
    
    # --- Percentiles inter-étapes (P95) ---
    interstage_rx_to_cpu_gpu_p95_ms: float  # RX → CPU-to-GPU P95
    interstage_cpu_gpu_to_proc_p95_ms: float # CPU-to-GPU → PROC(GPU) P95
    interstage_proc_to_gpu_cpu_p95_ms: float # PROC(GPU) → GPU-to-CPU P95
    interstage_gpu_cpu_to_tx_p95_ms: float  # GPU-to-CPU → TX P95
    
    # --- Nombre d'échantillons inter-étapes ---
    interstage_samples: int                  # nombre d'échantillons de métriques inter-étapes disponibles


@dataclass(frozen=True, slots = True)
class GatewayStatsSnapshot:
    """Immutable snapshot of gateway stats.

    Fields mirror the legacy snapshot() keys plus new latency and timing
    information. Use :meth:`to_dict` to obtain a backwards-compatible mapping.
    """

    # --- Débits instantanés ---
    fps_rx: float          # fréquence instantanée de réception (frames/s) – mesurée sur la dernière frame reçue
    fps_tx: float          # fréquence instantanée d’envoi (frames/s) – cadence actuelle de transmission vers Slicer

    # --- Moyennes glissantes ---
    avg_fps_rx: float      # débit moyen d’entrée (moyenne des N dernières mesures dans _rolling_fps_rx)
    avg_fps_tx: float      # débit moyen de sortie (moyenne des N dernières mesures dans _rolling_fps_tx)

    # --- Compteurs d’octets ---
    bytes_rx: int          # taille en octets du dernier message reçu (image ou tracking IGTLink)
    bytes_tx: int          # taille en octets du dernier message envoyé (masque + métadonnées)
    total_bytes_rx: int    # cumul total d’octets reçus depuis le démarrage du service
    total_bytes_tx: int    # cumul total d’octets envoyés depuis le démarrage du service

    # --- Horodatages d’activité ---
    last_rx_ts: float      # timestamp (epoch) du dernier message reçu (provenant du flux PlusServer)
    last_update_rx: float  # instant (time.time) où update_rx() a été appelé pour la dernière fois
    last_update_tx: float  # instant (time.time) où update_tx() a été appelé pour la dernière fois

    # --- Données temporelles globales ---
    started_at: float      # timestamp du démarrage du module GatewayStats (moment d’initialisation)
    uptime_s: float        # durée écoulée depuis started_at (temps de fonctionnement total, en secondes)

    # --- Statistiques de latence RX→TX ---
    latency_ms_avg: float  # latence moyenne RX→TX (en millisecondes) calculée sur les derniers échantillons
    latency_ms_p95: float  # latence au 95e percentile – 95 % des frames ont une latence inférieure à cette valeur
    latency_ms_max: float  # latence maximale observée (pic de latence récent)
    latency_samples: int   # nombre d’échantillons de latence actuellement conservés dans le buffer (max ≈ 200)

    # --- Indicateurs d'anomalie et de suivi ---
    latency_orphans: int   # nombre de frames TX sans RX correspondant (frames "orphelines" – ID manquant ou drop)
    snapshot_count: int    # compteur global du nombre de snapshots générés (permet de tracer la fréquence de reporting)

    # 🎯 ═══════════════════════════════════════════════════════════════════
    # NOUVELLES MÉTRIQUES INTER-ÉTAPES (Workflow GPU-résident détaillé)
    # ═══════════════════════════════════════════════════════════════════
    
    # --- Latences inter-étapes moyennes (en ms) ---
    interstage_rx_to_cpu_gpu_ms: float      # RX → CPU-to-GPU transfer latency
    interstage_cpu_gpu_to_proc_ms: float    # CPU-to-GPU → PROC(GPU) latency
    interstage_proc_to_gpu_cpu_ms: float    # PROC(GPU) → GPU-to-CPU transfer latency
    interstage_gpu_cpu_to_tx_ms: float      # GPU-to-CPU → TX latency
    
    # --- Percentiles inter-étapes (P95) ---
    interstage_rx_to_cpu_gpu_p95_ms: float  # RX → CPU-to-GPU P95
    interstage_cpu_gpu_to_proc_p95_ms: float # CPU-to-GPU → PROC(GPU) P95
    interstage_proc_to_gpu_cpu_p95_ms: float # PROC(GPU) → GPU-to-CPU P95
    interstage_gpu_cpu_to_tx_p95_ms: float  # GPU-to-CPU → TX P95
    
    # --- Nombre d'échantillons inter-étapes ---
    interstage_samples: int                  # nombre d'échantillons de métriques inter-étapes disponibles

    def to_dict(self) -> GatewayStatsDict:
        """Convertit le snapshot en dictionnaire standard pour la journalisation ou la sérialisation.

        Retourne un dictionnaire typé (`GatewayStatsDict`) contenant toutes les
        métriques actuelles du module GatewayStats. Utilisé par le manager et le
        superviseur pour exporter l’état courant de la passerelle.
        """
        return dict(
            # --- Débits instantanés ---
            fps_rx=self.fps_rx,           # fréquence instantanée de réception (frames/s)
            fps_tx=self.fps_tx,           # fréquence instantanée d’envoi (frames/s)
            # --- Moyennes glissantes ---
            avg_fps_rx=self.avg_fps_rx,   # débit moyen d’entrée – moyenne glissante sur N dernières frames
            avg_fps_tx=self.avg_fps_tx,   # débit moyen de sortie – moyenne glissante sur N dernières frames
            # --- Compteurs d’octets ---
            bytes_rx=self.bytes_rx,           # taille (octets) du dernier message reçu
            bytes_tx=self.bytes_tx,           # taille (octets) du dernier message envoyé
            total_bytes_rx=self.total_bytes_rx,   # cumul total d’octets reçus depuis le démarrage
            total_bytes_tx=self.total_bytes_tx,   # cumul total d’octets envoyés depuis le démarrage
            # --- Horodatages d’activité ---
            last_rx_ts=self.last_rx_ts,         # timestamp du dernier message reçu (flux PlusServer)
            last_update_rx=self.last_update_rx, # instant du dernier appel à update_rx() (activité RX)
            last_update_tx=self.last_update_tx, # instant du dernier appel à update_tx() (activité TX)
            # --- Données temporelles globales ---
            started_at=self.started_at,     # timestamp de création du module GatewayStats (démarrage)
            uptime_s=self.uptime_s,         # temps écoulé depuis started_at (durée de fonctionnement)
            # --- Statistiques de latence RX→TX ---
            latency_ms_avg=self.latency_ms_avg,   # latence moyenne RX→TX en millisecondes
            latency_ms_p95=self.latency_ms_p95,   # latence au 95e percentile (95 % des frames plus rapides)
            latency_ms_max=self.latency_ms_max,   # latence maximale observée sur la fenêtre récente
            latency_samples=self.latency_samples, # nombre d’échantillons utilisés pour les statistiques de latence
            # --- Indicateurs d'anomalie et de suivi ---
            latency_orphans=self.latency_orphans, # frames TX sans RX correspondant (frames "orphelines")
            snapshot_count=self.snapshot_count,   # compteur global de snapshots générés depuis le démarrage
            # 🎯 --- Métriques inter-étapes (Workflow GPU-résident détaillé) ---
            interstage_rx_to_cpu_gpu_ms=self.interstage_rx_to_cpu_gpu_ms,         # RX → CPU-to-GPU latency moyenne
            interstage_cpu_gpu_to_proc_ms=self.interstage_cpu_gpu_to_proc_ms,     # CPU-to-GPU → PROC(GPU) latency moyenne
            interstage_proc_to_gpu_cpu_ms=self.interstage_proc_to_gpu_cpu_ms,     # PROC(GPU) → GPU-to-CPU latency moyenne
            interstage_gpu_cpu_to_tx_ms=self.interstage_gpu_cpu_to_tx_ms,         # GPU-to-CPU → TX latency moyenne
            interstage_rx_to_cpu_gpu_p95_ms=self.interstage_rx_to_cpu_gpu_p95_ms, # RX → CPU-to-GPU P95
            interstage_cpu_gpu_to_proc_p95_ms=self.interstage_cpu_gpu_to_proc_p95_ms, # CPU-to-GPU → PROC(GPU) P95
            interstage_proc_to_gpu_cpu_p95_ms=self.interstage_proc_to_gpu_cpu_p95_ms, # PROC(GPU) → GPU-to-CPU P95
            interstage_gpu_cpu_to_tx_p95_ms=self.interstage_gpu_cpu_to_tx_p95_ms, # GPU-to-CPU → TX P95
            interstage_samples=self.interstage_samples,                            # nombre d'échantillons inter-étapes
        )




class GatewayStats:
    """Collecteur de statistiques thread-safe pour la passerelle.

    Méthodes publiques (compatibles avec les versions précédentes) :
      - update_rx(fps, ts, bytes_count=0)     → met à jour les métriques de réception
      - update_tx(fps, bytes_count=0)         → met à jour les métriques d’envoi
      - snapshot() -> GatewayStatsDict        → retourne un instantané typé des statistiques actuelles
      - set_last_rx_ts(ts)                    → met à jour le timestamp du dernier message reçu

    Nouvelles méthodes :
      - mark_rx(frame_id, rx_ts)              → enregistre l’heure de réception d’une frame
      - mark_tx(frame_id, tx_ts)              → enregistre l’heure d’envoi correspondante et calcule la latence
      - reset(cold_start: bool = False)       → réinitialise les compteurs et buffers (optionnellement redémarre le chronomètre global)

    Sécurité multi-thread :
      Un verrou interne unique (_lock) protège l’ensemble des états mutables.
      Toutes les opérations sont en moyenne en O(1). Les fenêtres glissantes et
      les buffers de latence sont bornés pour éviter toute fuite mémoire.
    """

    def __init__(self, rolling_window_size: int = 20, latency_window_size: int = 200) -> None:
        """Initialise le module GatewayStats.

        Args:
            rolling_window_size: nombre d’échantillons récents du débit (fps) à utiliser pour le calcul de la moyenne (minimum : 3).
            latency_window_size: nombre maximal d’échantillons de latence à conserver dans le buffer (valeur bornée pour éviter les surcharges mémoire).
        """
        self._lock = threading.Lock()                 # verrou interne pour assurer la sécurité multi-thread lors des accès aux métriques

        # --- Champs de fréquence instantanée (fps) ---
        self._fps_rx: float = 0.0                     # fréquence instantanée de réception (frames/s) – mise à jour à chaque frame reçue
        self._fps_tx: float = 0.0                     # fréquence instantanée d’envoi (frames/s) – mise à jour à chaque frame transmise
        self._last_rx_ts: float = 0.0                 # timestamp (epoch) de la dernière frame reçue
        self._last_update_rx: float = 0.0             # instant (time.time) du dernier appel à update_rx()
        self._last_update_tx: float = 0.0             # instant (time.time) du dernier appel à update_tx()

        # --- Compteurs d’octets ---
        self._bytes_rx: int = 0                       # taille du dernier message reçu en octets
        self._bytes_tx: int = 0                       # taille du dernier message envoyé en octets
        self._total_bytes_rx: int = 0                 # nombre total d’octets reçus depuis le démarrage du service
        self._total_bytes_tx: int = 0                 # nombre total d’octets envoyés depuis le démarrage du service

        # --- Fenêtres glissantes (rolling windows) ---
        if rolling_window_size < 3:                   # sécurité : impose une taille minimale pour éviter les moyennes instables
            LOG.warning("rolling_window_size < 3, forcing to 3")
            rolling_window_size = 3
        self._rolling_window_size = int(rolling_window_size)  # taille effective de la fenêtre glissante utilisée pour le calcul moyen du fps
        self._rolling_fps_rx: Deque[float] = collections.deque(maxlen=self._rolling_window_size)  # historique borné des fps_rx récents
        self._rolling_fps_tx: Deque[float] = collections.deque(maxlen=self._rolling_window_size)  # historique borné des fps_tx récents

        # --- Temps de démarrage global ---
        self._started_at: float = time.time()          # horodatage du démarrage du module GatewayStats (référence pour le calcul d’uptime)

        # --- Mesure de latence RX→TX ---
        self._latency_window_size = int(latency_window_size)  # nombre maximal d’échantillons de latence conservés dans le buffer
        if self._latency_window_size <= 0:             # vérification de validité – évite les tailles nulles ou négatives
            self._latency_window_size = 200
        self._latency_buffer_ms: Deque[float] = collections.deque(maxlen=self._latency_window_size)  # buffer borné des latences en millisecondes
        self._pending_rx: Dict[int, float] = {}        # table temporaire frame_id → rx_ts pour calculer les latences lors du mark_tx()

        # --- Compteurs optionnels de diagnostic ---
        self._latency_orphans: int = 0                 # nombre de frames TX sans RX correspondant (frames “orphelines”)
        self._last_prune_time: float = time.time()     # dernier instant où les entrées obsolètes de _pending_rx ont été nettoyées

        # --- Compteur global de snapshots ---
        self._snapshot_count: int = 0                  # nombre total de snapshots générés (utile pour le suivi ou les logs)

        # 🎯 ═══════════════════════════════════════════════════════════════════
        # NOUVELLES MÉTRIQUES INTER-ÉTAPES (Workflow GPU-résident détaillé)
        # ═══════════════════════════════════════════════════════════════════
        
        # --- Buffers pour métriques inter-étapes ---
        self._interstage_window_size = int(latency_window_size)  # même taille que latency_window_size
        self._interstage_rx_to_cpu_gpu: Deque[float] = collections.deque(maxlen=self._interstage_window_size)
        self._interstage_cpu_gpu_to_proc: Deque[float] = collections.deque(maxlen=self._interstage_window_size)
        self._interstage_proc_to_gpu_cpu: Deque[float] = collections.deque(maxlen=self._interstage_window_size)
        self._interstage_gpu_cpu_to_tx: Deque[float] = collections.deque(maxlen=self._interstage_window_size)
        
        # --- Stockage temporaire des timestamps inter-étapes ---
        self._pending_interstage: Dict[int, Dict[str, float]] = {}  # frame_id → {t_rx, t1, t2, t3, t_tx}


    # ---------------------- legacy API (preserved) ----------------------
    def update_rx(self, fps: float, ts: float, bytes_count: int = 0) -> None:
        """Met à jour les compteurs de réception (fréquence et taille en octets).

        Args:
            fps: fréquence mesurée en images par seconde (frames-per-second).
            ts: horodatage (en secondes epoch) de la dernière image reçue.
            bytes_count: taille en octets du dernier message reçu.

        Sécurité multi-thread :
            Acquiert le verrou interne (_lock) pendant la mise à jour.
        Complexité :
            O(1) amorti (append/pop dans une deque bornée).
        """
        with self._lock:                                   # verrouille la section critique pour éviter les conflits entre threads
            self._fps_rx = float(fps)                      # enregistre la fréquence instantanée de réception
            self._last_rx_ts = float(ts)                   # mémorise le timestamp de la dernière frame reçue
            self._last_update_rx = time.time()             # stocke le moment exact de la mise à jour (référence interne)
            self._bytes_rx = int(bytes_count)              # met à jour la taille du dernier paquet reçu en octets
            self._total_bytes_rx += int(bytes_count)       # incrémente le total cumulé d’octets reçus depuis le démarrage
            self._rolling_fps_rx.append(float(fps))        # ajoute la fréquence actuelle dans la fenêtre glissante (historique borné)
        from core.monitoring.kpi import format_kpi, safe_log_kpi, is_kpi_enabled  # import local pour éviter dépendance circulaire globale
        if is_kpi_enabled():                               # vérifie si la journalisation KPI est activée (via variable d'environnement)
            try:
                msg = format_kpi({                         # construit un message KPI structuré de type key=value
                    "ts": time.time(),                     # timestamp actuel (epoch, haute précision)
                    "event": "rx_update",                  # identifiant de l'événement pour l'analyse ultérieure
                    "fps_rx": self._fps_rx,                # fréquence instantanée de réception (frames/s)
                    "bytes_rx": self._bytes_rx,            # taille du dernier message reçu (octets)
                    "total_bytes_rx": self._total_bytes_rx # cumul total d’octets reçus depuis le démarrage
                })
                safe_log_kpi(msg)                          # envoie la ligne formatée vers le logger KPI (écriture asynchrone)
            except Exception:                              # sécurité : ne jamais propager une erreur de log vers le flux principal
                LOG.debug("KPI emission failed in GatewayStats.update_rx()")  # trace discrète en cas d’échec du log KPI

    def update_tx(self, fps: float, bytes_count: int = 0) -> None:
        """Met à jour les compteurs d’envoi (fréquence et taille en octets).

        Remarque :
            La signature de la méthode est conservée pour compatibilité avec les versions précédentes.
        """
        with self._lock:                                   # verrouille la section critique pour garantir la cohérence des données multi-thread
            self._fps_tx = float(fps)                      # enregistre la fréquence instantanée d’envoi (frames/s)
            self._last_update_tx = time.time()             # mémorise le moment exact de la dernière mise à jour (référence interne)
            self._bytes_tx = int(bytes_count)              # met à jour la taille du dernier message envoyé en octets
            self._total_bytes_tx += int(bytes_count)       # incrémente le total cumulé d’octets envoyés depuis le démarrage du service
            self._rolling_fps_tx.append(float(fps))        # ajoute la fréquence actuelle dans la fenêtre glissante (historique borné pour la moyenne)
            
        from core.monitoring.kpi import format_kpi, safe_log_kpi, is_kpi_enabled  # import local pour éviter dépendance circulaire globale
        if is_kpi_enabled():                               # vérifie si la journalisation KPI est activée (contrôle via variable d'environnement)
            try:
                msg = format_kpi({                         # construit un message KPI structuré key=value
                    "ts": time.time(),                     # timestamp actuel (epoch)
                    "event": "tx_update",                  # identifiant de l'événement (utile pour filtrage dans les logs)
                    "fps_tx": self._fps_tx,                # fréquence instantanée d’envoi (frames/s)
                    "bytes_tx": self._bytes_tx,            # taille du dernier message envoyé (octets)
                    "total_bytes_tx": self._total_bytes_tx # cumul total d’octets envoyés depuis le démarrage
                })
                safe_log_kpi(msg)                          # envoie le message formaté vers le logger KPI (asynchrone via QueueHandler)
            except Exception:                              # bloc de sécurité pour garantir robustesse du flux principal
                LOG.debug("KPI emission failed in GatewayStats.update_tx()")  # log discret en cas d’erreur d’écriture KPI


    def snapshot(self) -> GatewayStatsDict:
        """Retourne un instantané typé des statistiques actuelles (compatibilité ascendante).

        Description :
            Construit une vue figée de l’état courant des compteurs RX/TX, des
            moyennes glissantes et des mesures de latence. Le dictionnaire
            retourné contient à la fois les anciennes clés (historiques) et les
            nouvelles (started_at, uptime_s, latence…). Les anciens consommateurs
            peuvent ignorer les clés supplémentaires sans erreur.
        """
        with self._lock:                                                   # verrouille l’accès concurrent aux données pendant la lecture du snapshot
            now = time.time()                                              # capture le temps actuel pour le calcul de l’uptime et des deltas temporels

            # --- Calcul des moyennes glissantes (fps) ---
            avg_rx = float(sum(self._rolling_fps_rx) / len(self._rolling_fps_rx)) if self._rolling_fps_rx else self._fps_rx
            # calcule le débit moyen d’entrée (fps_rx) sur la fenêtre glissante, ou la valeur instantanée si vide
            avg_tx = float(sum(self._rolling_fps_tx) / len(self._rolling_fps_tx)) if self._rolling_fps_tx else self._fps_tx
            # calcule le débit moyen de sortie (fps_tx) sur la fenêtre glissante, ou la valeur instantanée si vide

            # --- Calcul des statistiques de latence RX→TX ---
            lat_list: List[float] = list(self._latency_buffer_ms)          # convertit le buffer de latences en liste pour traitement
            latency_samples = len(lat_list)                                # nombre total d’échantillons actuellement disponibles

            if latency_samples:                                            # si le buffer contient des valeurs de latence
                latency_ms_avg = float(sum(lat_list) / latency_samples)    # calcule la latence moyenne (en millisecondes)
                latency_ms_max = float(max(lat_list))                      # extrait la latence maximale observée

                # --- Calcul du 95e percentile ---
                # Utilise NumPy si disponible (plus rapide), sinon calcul manuel
                if _HAS_NUMPY and latency_samples >= 5:                    # si assez d’échantillons et NumPy disponible
                    try:
                        latency_ms_p95 = float(np.percentile(lat_list, 95))# calcule le 95e percentile avec NumPy
                    except Exception:
                        sorted_lat = sorted(lat_list)                      # repli : tri manuel
                        idx = max(0, int(0.95 * (latency_samples - 1)))    # index correspondant au 95e percentile
                        latency_ms_p95 = float(sorted_lat[idx])            # extraction manuelle du percentile
                else:                                                      # si NumPy absent ou échantillons insuffisants
                    sorted_lat = sorted(lat_list)                          # tri manuel des latences
                    idx = max(0, int(0.95 * (latency_samples - 1)))        # index du 95e percentile
                    latency_ms_p95 = float(sorted_lat[idx])                # valeur du percentile 95 manuelle
            else:
                # cas où aucune mesure de latence n'est encore disponible
                latency_ms_avg = 0.0                                       # latence moyenne par défaut
                latency_ms_p95 = 0.0                                       # latence au 95e percentile par défaut
                latency_ms_max = 0.0                                       # latence maximale par défaut

            # 🎯 --- Calcul des statistiques inter-étapes (Workflow GPU-résident détaillé) ---
            rx_to_cpu_gpu_avg, rx_to_cpu_gpu_p95 = self._calculate_interstage_stats(self._interstage_rx_to_cpu_gpu)
            cpu_gpu_to_proc_avg, cpu_gpu_to_proc_p95 = self._calculate_interstage_stats(self._interstage_cpu_gpu_to_proc)
            proc_to_gpu_cpu_avg, proc_to_gpu_cpu_p95 = self._calculate_interstage_stats(self._interstage_proc_to_gpu_cpu)
            gpu_cpu_to_tx_avg, gpu_cpu_to_tx_p95 = self._calculate_interstage_stats(self._interstage_gpu_cpu_to_tx)
            interstage_samples_count = min(len(self._interstage_rx_to_cpu_gpu), len(self._interstage_cpu_gpu_to_proc),
                                         len(self._interstage_proc_to_gpu_cpu), len(self._interstage_gpu_cpu_to_tx))

            # --- Incrément du compteur global de snapshots ---
            self._snapshot_count += 1                                      # incrémente le nombre total d’instantanés générés

            # --- Construction de l’objet immuable GatewayStatsSnapshot ---
            snap = GatewayStatsSnapshot(
                fps_rx=self._fps_rx,                                       # fréquence instantanée de réception
                fps_tx=self._fps_tx,                                       # fréquence instantanée d’envoi
                avg_fps_rx=avg_rx,                                         # débit moyen RX (moyenne glissante)
                avg_fps_tx=avg_tx,                                         # débit moyen TX (moyenne glissante)
                bytes_rx=self._bytes_rx,                                   # taille du dernier message reçu
                bytes_tx=self._bytes_tx,                                   # taille du dernier message envoyé
                total_bytes_rx=self._total_bytes_rx,                       # octets reçus cumulés depuis le démarrage
                total_bytes_tx=self._total_bytes_tx,                       # octets envoyés cumulés depuis le démarrage
                last_rx_ts=self._last_rx_ts,                               # timestamp de la dernière frame reçue
                last_update_rx=self._last_update_rx,                       # instant du dernier update_rx()
                last_update_tx=self._last_update_tx,                       # instant du dernier update_tx()
                started_at=self._started_at,                               # timestamp du démarrage du module GatewayStats
                uptime_s=max(0.0, now - self._started_at),                 # temps écoulé depuis le démarrage (en secondes)
                latency_ms_avg=latency_ms_avg,                             # latence moyenne RX→TX
                latency_ms_p95=latency_ms_p95,                             # latence au 95e percentile
                latency_ms_max=latency_ms_max,                             # latence maximale récente
                latency_samples=latency_samples,                           # nombre d'échantillons de latence utilisés
                latency_orphans=self._latency_orphans,                     # nombre de frames TX sans RX correspondant
                snapshot_count=self._snapshot_count,                       # numéro séquentiel du snapshot
                # 🎯 Métriques inter-étapes (Workflow GPU-résident détaillé)
                interstage_rx_to_cpu_gpu_ms=rx_to_cpu_gpu_avg,            # RX → CPU-to-GPU latency moyenne
                interstage_cpu_gpu_to_proc_ms=cpu_gpu_to_proc_avg,        # CPU-to-GPU → PROC(GPU) latency moyenne
                interstage_proc_to_gpu_cpu_ms=proc_to_gpu_cpu_avg,        # PROC(GPU) → GPU-to-CPU latency moyenne
                interstage_gpu_cpu_to_tx_ms=gpu_cpu_to_tx_avg,            # GPU-to-CPU → TX latency moyenne
                interstage_rx_to_cpu_gpu_p95_ms=rx_to_cpu_gpu_p95,        # RX → CPU-to-GPU P95
                interstage_cpu_gpu_to_proc_p95_ms=cpu_gpu_to_proc_p95,    # CPU-to-GPU → PROC(GPU) P95
                interstage_proc_to_gpu_cpu_p95_ms=proc_to_gpu_cpu_p95,    # PROC(GPU) → GPU-to-CPU P95
                interstage_gpu_cpu_to_tx_p95_ms=gpu_cpu_to_tx_p95,        # GPU-to-CPU → TX P95
                interstage_samples=interstage_samples_count,               # nombre d'échantillons inter-étapes
            )

            # attach optional drop counters from global KPI module if present
            out = snap.to_dict()
            try:
                from core.monitoring.kpi import get_drop_count

                out["drops_rx_total"] = int(get_drop_count("rx.drop_total") or 0)
                out["drops_tx_total"] = int(get_drop_count("tx.drop_total") or 0)
            except Exception:
                out["drops_rx_total"] = 0
                out["drops_tx_total"] = 0

            # alias average latency field for compatibility
            try:
                out["avg_latency_ms"] = float(out.get("latency_ms_avg", 0.0))
            except Exception:
                out["avg_latency_ms"] = 0.0

            return out  # convertit l’objet immuable en dictionnaire typé pour export (GatewayStatsDict)

    def set_last_rx_ts(self, ts: float) -> None:
        """Met à jour de manière sûre le timestamp du dernier message reçu.

        Args:
            ts: horodatage (en secondes epoch) de la dernière frame reçue.
        """
        with self._lock:                            # verrouille la section critique pour éviter les accès concurrents
            self._last_rx_ts = float(ts)             # enregistre le timestamp reçu comme dernier instant de réception
            self._last_update_rx = time.time()       # mémorise le moment exact de la mise à jour (utilisé pour détecter une inactivité RX)

    # ---------------------- API de latence ----------------------
    def mark_rx(self, frame_id: int, rx_ts: float) -> None:
        """Enregistre qu’une frame d’identifiant `frame_id` a été reçue à l’instant `rx_ts`.

        Args:
            frame_id: identifiant unique de la frame (entier).
            rx_ts: horodatage de réception en secondes (généralement time.time()).

        Comportement :
            Stocke le timestamp RX dans un dictionnaire temporaire jusqu’à l’appel
            ultérieur de `mark_tx()`. Ce mapping est nettoyé périodiquement pour
            éviter toute fuite mémoire. Opération O(1).
        """
        try:
            fid = int(frame_id)                      # force le type entier pour s’assurer d’un identifiant valide
        except Exception:
            return                                   # en cas d’erreur de conversion, abandon silencieux
        with self._lock:                             # verrouillage pour protéger le dictionnaire partagé
            self._pending_rx[fid] = float(rx_ts)     # associe le frame_id à son timestamp de réception
            if LOG.isEnabledFor(logging.DEBUG):      
                LOG.debug("Registered RX frame_id=%s at ts=%.3f", fid, rx_ts)  # log facultatif pour le debug détaillé
            # nettoyage périodique des anciennes entrées RX
            if len(self._pending_rx) % 100 == 0:     
                try:
                    self._prune_pending_rx()         # supprime les entrées obsolètes si la table devient volumineuse
                except Exception:
                    LOG.debug("_prune_pending_rx failed")  # log discret en cas d’échec du nettoyage

    def mark_tx(self, frame_id: int, tx_ts: float) -> None:
        """Enregistre qu’une frame d’identifiant `frame_id` a été transmise à l’instant `tx_ts`.

        Comportement :
            Si un timestamp RX correspondant existe, calcule la latence (ms) et
            l’ajoute au buffer borné. Les transmissions sans correspondance RX
            sont comptabilisées comme “orphelines”.
        """
        try:
            fid = int(frame_id)                      # conversion stricte de l’identifiant pour cohérence
        except Exception:
            return                                   # si l’ID est invalide, on ignore silencieusement
        with self._lock:                             # verrouillage pour opérations thread-safe
            rx_ts = self._pending_rx.pop(fid, None)  # récupère et retire le timestamp RX correspondant (s’il existe)
            if rx_ts is None:                        # cas d’absence : frame TX orpheline (pas de RX préalable)
                self._latency_orphans += 1           # incrémente le compteur d’anomalies
                return
            latency_ms = (float(tx_ts) - float(rx_ts)) * 1000.0  # calcule la latence en millisecondes
            if latency_ms < 0.0:                     # sécurité : si le TX est antérieur au RX (anomalie)
                LOG.warning("mark_tx: tx_ts < rx_ts for frame_id=%s (clamped to 0)", fid)
                latency_ms = 0.0                     # on force la latence à zéro
            self._latency_buffer_ms.append(float(latency_ms))    # stocke la latence dans le buffer borné
            if LOG.isEnabledFor(logging.DEBUG):      
                LOG.debug("Frame %s latency=%.2f ms", fid, latency_ms)  # log de diagnostic en mode DEBUG
            # nettoyage périodique du dictionnaire RX
            if len(self._pending_rx) % 100 == 0:     
                try:
                    self._prune_pending_rx()         # supprime les entrées RX trop anciennes
                except Exception:
                    LOG.debug("_prune_pending_rx failed")

    def _prune_pending_rx(self, max_age_s: float = 5.0) -> None:
        """Supprime les entrées RX âgées de plus de `max_age_s` secondes (frames périmées)."""
        now = time.time()                            # temps actuel pour calculer l’âge des entrées
        removed = 0                                  # compteur du nombre d’éléments supprimés
        keys_to_remove = [k for k, ts in self._pending_rx.items() if now - ts > max_age_s]
        # génère la liste des clés à supprimer (frames trop anciennes)
        for k in keys_to_remove:
            try:
                del self._pending_rx[k]              # supprime chaque entrée obsolète
                removed += 1
            except KeyError:
                pass                                 # sécurité : ignore si déjà supprimée entre-temps
        if removed and LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Pruned %d stale pending_rx entries", removed)  # log informatif du nettoyage effectué

    # 🎯 ═══════════════════════════════════════════════════════════════════
    # NOUVELLES MÉTHODES INTER-ÉTAPES (Workflow GPU-résident détaillé)
    # ═══════════════════════════════════════════════════════════════════

    def mark_interstage_rx(self, frame_id: int, t_rx: float) -> None:
        """Enregistre le début du workflow inter-étapes (timestamp RX).
        
        Args:
            frame_id: identifiant unique de la frame
            t_rx: timestamp de réception RX
        """
        try:
            fid = int(frame_id)
        except Exception:
            return
        with self._lock:
            if fid not in self._pending_interstage:
                self._pending_interstage[fid] = {}
            self._pending_interstage[fid]['t_rx'] = float(t_rx)

    def mark_interstage_cpu_to_gpu(self, frame_id: int, t1: float) -> None:
        """Enregistre la fin du transfert CPU→GPU (t1).
        
        Args:
            frame_id: identifiant unique de la frame
            t1: timestamp de fin du transfert CPU→GPU
        """
        try:
            fid = int(frame_id)
        except Exception:
            return
        with self._lock:
            if fid not in self._pending_interstage:
                self._pending_interstage[fid] = {}
            self._pending_interstage[fid]['t1'] = float(t1)

    def mark_interstage_proc_done(self, frame_id: int, t2: float) -> None:
        """Enregistre la fin du processing GPU (t2).
        
        Args:
            frame_id: identifiant unique de la frame
            t2: timestamp de fin du processing GPU
        """
        try:
            fid = int(frame_id)
        except Exception:
            return
        with self._lock:
            if fid not in self._pending_interstage:
                self._pending_interstage[fid] = {}
            self._pending_interstage[fid]['t2'] = float(t2)

    def mark_interstage_gpu_to_cpu(self, frame_id: int, t3: float) -> None:
        """Enregistre la fin du transfert GPU→CPU (t3).
        
        Args:
            frame_id: identifiant unique de la frame  
            t3: timestamp de fin du transfert GPU→CPU
        """
        try:
            fid = int(frame_id)
        except Exception:
            return
        with self._lock:
            if fid not in self._pending_interstage:
                self._pending_interstage[fid] = {}
            self._pending_interstage[fid]['t3'] = float(t3)

    def mark_interstage_tx(self, frame_id: int, t_tx: float) -> None:
        """Enregistre la fin du workflow (timestamp TX) et calcule toutes les latences inter-étapes.
        
        Args:
            frame_id: identifiant unique de la frame
            t_tx: timestamp TX final
        """
        try:
            fid = int(frame_id)
        except Exception:
            return
        
        with self._lock:
            if fid not in self._pending_interstage:
                return  # Pas de données inter-étapes pour cette frame
            
            stages = self._pending_interstage[fid]
            stages['t_tx'] = float(t_tx)
            
            # Calcul des latences inter-étapes (en ms) si tous les timestamps sont présents
            if all(k in stages for k in ['t_rx', 't1', 't2', 't3', 't_tx']):
                # RX → CPU-to-GPU
                rx_to_cpu_gpu_ms = (stages['t1'] - stages['t_rx']) * 1000.0
                # CPU-to-GPU → PROC(GPU)  
                cpu_gpu_to_proc_ms = (stages['t2'] - stages['t1']) * 1000.0
                # PROC(GPU) → GPU-to-CPU
                proc_to_gpu_cpu_ms = (stages['t3'] - stages['t2']) * 1000.0
                # GPU-to-CPU → TX
                gpu_cpu_to_tx_ms = (stages['t_tx'] - stages['t3']) * 1000.0
                
                # Stockage des métriques dans les buffers
                self._interstage_rx_to_cpu_gpu.append(rx_to_cpu_gpu_ms)
                self._interstage_cpu_gpu_to_proc.append(cpu_gpu_to_proc_ms)
                self._interstage_proc_to_gpu_cpu.append(proc_to_gpu_cpu_ms)
                self._interstage_gpu_cpu_to_tx.append(gpu_cpu_to_tx_ms)
                
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Interstage latencies frame_id=%s: RX→GPU=%.2f, GPU→PROC=%.2f, PROC→CPU=%.2f, CPU→TX=%.2f ms", 
                             fid, rx_to_cpu_gpu_ms, cpu_gpu_to_proc_ms, proc_to_gpu_cpu_ms, gpu_cpu_to_tx_ms)
            
            # Nettoyage de l'entrée traitée
            del self._pending_interstage[fid]
            
            # Nettoyage périodique
            if len(self._pending_interstage) % 50 == 0:
                self._prune_pending_interstage()

    def _prune_pending_interstage(self, max_age_s: float = 5.0) -> None:
        """Supprime les entrées inter-étapes âgées de plus de max_age_s secondes."""
        now = time.time()
        removed = 0
        keys_to_remove = []
        
        for fid, stages in self._pending_interstage.items():
            # Utilise t_rx comme référence temporelle
            if 't_rx' in stages and now - stages['t_rx'] > max_age_s:
                keys_to_remove.append(fid)
        
        for k in keys_to_remove:
            try:
                del self._pending_interstage[k]
                removed += 1
            except KeyError:
                pass
        
        if removed and LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Pruned %d stale pending_interstage entries", removed)

    def _calculate_interstage_stats(self, buffer: Deque[float]) -> tuple[float, float]:
        """Calcule moyenne et P95 pour un buffer de métriques inter-étapes.
        
        Returns:
            tuple: (moyenne_ms, p95_ms)
        """
        if not buffer:
            return 0.0, 0.0
        
        data = list(buffer)
        avg_ms = sum(data) / len(data)
        
        # Calcul P95
        if len(data) >= 2:
            sorted_data = sorted(data)
            p95_idx = int(0.95 * len(sorted_data))
            p95_ms = sorted_data[min(p95_idx, len(sorted_data) - 1)]
        else:
            p95_ms = avg_ms
        
        return avg_ms, p95_ms

    # ---------------------- Gestion générale ----------------------
    def reset(self, cold_start: bool = False) -> None:
        """Réinitialise les compteurs et buffers volatils.

        Args:
            cold_start: si True, réinitialise également le timestamp de démarrage
                        (réinitialise ainsi l’uptime).
        
        Description :
            Efface toutes les fenêtres glissantes, les buffers de latence, les
            compteurs d’octets et les timestamps récents. Le champ started_at
            est conservé sauf en cas de redémarrage complet (cold_start=True).
        """
        with self._lock:                             # verrouille pour éviter les mises à jour concurrentes
            self._fps_rx = 0.0                       # remet à zéro le fps instantané de réception
            self._fps_tx = 0.0                       # remet à zéro le fps instantané d’envoi
            self._last_rx_ts = 0.0                   # réinitialise le timestamp du dernier RX
            self._last_update_rx = 0.0               # réinitialise l’heure du dernier update_rx()
            self._last_update_tx = 0.0               # réinitialise l’heure du dernier update_tx()
            self._bytes_rx = 0                       # réinitialise le compteur d’octets reçus (frame courante)
            self._bytes_tx = 0                       # réinitialise le compteur d’octets envoyés (frame courante)
            self._total_bytes_rx = 0                 # remet à zéro le total cumulé d’octets reçus
            self._total_bytes_tx = 0                 # remet à zéro le total cumulé d’octets envoyés
            self._rolling_fps_rx.clear()             # vide la fenêtre glissante RX
            self._rolling_fps_tx.clear()             # vide la fenêtre glissante TX
            self._latency_buffer_ms.clear()          # efface le buffer de latence
            self._pending_rx.clear()                 # efface la table des frames en attente de TX
            self._latency_orphans = 0                # remet à zéro le compteur de frames orphelines
            # 🎯 Réinitialisation des métriques inter-étapes
            self._interstage_rx_to_cpu_gpu.clear()   # efface le buffer RX→CPU-to-GPU
            self._interstage_cpu_gpu_to_proc.clear() # efface le buffer CPU-to-GPU→PROC
            self._interstage_proc_to_gpu_cpu.clear() # efface le buffer PROC→GPU-to-CPU
            self._interstage_gpu_cpu_to_tx.clear()   # efface le buffer GPU-to-CPU→TX
            self._pending_interstage.clear()         # efface la table des timestamps inter-étapes
            if cold_start:                           
                self._started_at = time.time()       # redéfinit l'heure de démarrage si redémarrage complet demandé
