"""Monitoring thread et helpers (Thread M)

Description
-----------
Module chargé de surveiller la pipeline en temps réel :
- calcul et publication des métriques (fps_in, fps_out, latence E2E),
- relevé de l'utilisation GPU (si disponible),
- statistiques sur la taille/consommation des queues,
- ajustement de `Queue_RT_dyn` via `adaptive_queue_resize`.

Responsabilités
----------------
- démarrer un thread périodique qui collecte métriques,
- exposer une API pour journaliser des métriques ponctuelles,
- appliquer des règles d'ajustement sur les queues.

Dépendances attendues
---------------------
- accès aux queues (module `core.queues.buffers`),
- utilitaire de logging (configuration via `src/config/logging.yaml`),
- possibilité d'utiliser `pynvml` ou `nvidia-smi` pour récupération GPU (optionnel).

Fonctions principales
---------------------
- start_monitor_thread(config=None)
- log_pipeline_metrics(fps_in, fps_out, latency_ms, gpu_util, extras=None)
- adjust_rt_queue_size(current_size, metrics) -> int

Note
----
Les fonctions ci-dessous ne contiennent pas d'implémentation réelle —
elles servent de contrat pour l'implémentation ultérieure.
"""

from typing import Dict, Optional   # pour indiquer les types de retour (dictionnaire, optionnel)
import time                         # utilisé pour mesurer le temps et horodater les métriques
import logging                      # système de journalisation Python
import json                         # pour exporter les métriques en JSON
from pathlib import Path             # pour manipuler les chemins de fichiers (logs, snapshots)

LOG = logging.getLogger("igt.monitor")  # création d’un logger nommé "igt.monitor" (thread de monitoring)

from core.queues.buffers import collect_queue_metrics  # fonction utilitaire pour extraire les métriques des files (queues)
from collections import deque                          # structure de données (file circulaire) utilisée pour stocker un historique des métriques
from core.monitoring.kpi import safe_log_kpi, is_kpi_enabled  # fonctions utilitaires pour écrire les KPI en toute sécurité

# Petite mémoire tampon (deque) conservant un historique des métriques récentes (ex: pour calculer une moyenne glissante)
_METRICS_HISTORY = deque(maxlen=30)
# Nombre minimum de points requis avant de calculer une moyenne lissée (évite de calculer avec trop peu d'échantillons)
_SMOOTH_MIN_POINTS = 10

# 🎯 Référence globale au gateway actif pour récupération des vraies métriques
_ACTIVE_GATEWAY = None


def set_active_gateway(gateway):
    """Définit la référence au gateway actif pour la collecte de métriques.
    
    Args:
        gateway: Instance du gateway IGT ou None pour désactiver
    """
    global _ACTIVE_GATEWAY
    _ACTIVE_GATEWAY = gateway
    LOG.debug("Active gateway set: %s", gateway is not None)


def get_active_gateway():
    """Retourne la référence au gateway actif ou None si aucun gateway configuré."""
    return _ACTIVE_GATEWAY


def aggregate_metrics(fps_in: float, fps_out: float, latency_ms: float, gpu_util: float):
    """Ajoute un nouvel échantillon de métriques et retourne la moyenne lissée si suffisamment de données ont été accumulées."""
    _METRICS_HISTORY.append({  # ajoute un dictionnaire contenant les mesures actuelles dans l’historique
        "ts": time.time(),          # horodatage en secondes
        "fps_in": fps_in,           # fréquence d’images en entrée
        "fps_out": fps_out,         # fréquence d’images en sortie
        "latency_ms": latency_ms,   # latence moyenne en millisecondes
        "gpu_util": gpu_util,       # taux d’utilisation GPU en pourcentage
    })
    # Ne retourne une moyenne lissée que si on a un nombre suffisant d’échantillons
    if len(_METRICS_HISTORY) < _SMOOTH_MIN_POINTS:
        return None  # pas encore assez de points pour lisser
    avg = {  # calcul d’une moyenne simple pour chaque clé sur l’ensemble de l’historique
        k: sum(m[k] for m in _METRICS_HISTORY) / len(_METRICS_HISTORY)
        for k in ("fps_in", "fps_out", "latency_ms", "gpu_util")
    }
    LOG.debug("Smoothed KPIs: %s", avg)  # log interne pour indiquer les valeurs lissées calculées
    return avg  # renvoie le dictionnaire contenant les moyennes lissées


def get_aggregated_metrics(gateway=None) -> Optional[Dict[str, float]]:
    """Retourne la moyenne lissée actuelle calculée à partir de l'historique en mémoire.
    
    Si un gateway est fourni, récupère les vraies métriques du gateway et les fusionne
    avec les métriques simulées. Si aucun gateway n'est fourni, utilise le gateway actif global.

    Args:
        gateway: Instance optionnelle du gateway IGT pour récupérer les vraies métriques
                Si None, utilise le gateway actif global (_ACTIVE_GATEWAY)

    Returns:
        Dictionnaire des métriques agrégées ou None si insuffisamment de données
    """
    if len(_METRICS_HISTORY) < _SMOOTH_MIN_POINTS:  # si on n'a pas assez de données accumulées
        return None
    
    # Calcul de la moyenne lissée à partir de l'historique simulé
    avg = {  # même logique que aggregate_metrics, mais sans ajout de nouveau point
        k: sum(m[k] for m in _METRICS_HISTORY) / len(_METRICS_HISTORY)
        for k in ("fps_in", "fps_out", "latency_ms", "gpu_util")
    }
    
    # 🎯 FUSION avec les vraies métriques du gateway si disponible
    active_gw = gateway or _ACTIVE_GATEWAY
    if active_gw is not None:
        try:
            gw_metrics = collect_gateway_metrics(active_gw)
            if gw_metrics:
                # Remplace les métriques simulées par les vraies métriques du gateway
                avg.update({
                    "fps_rx": gw_metrics.get('fps_rx', avg.get('fps_in', 0.0)),
                    "fps_tx": gw_metrics.get('fps_tx', avg.get('fps_out', 0.0)),
                    "latency_ms": gw_metrics.get('avg_latency_ms', avg.get('latency_ms', 0.0)),
                    "bytes_rx_MB": gw_metrics.get('bytes_rx_MB', 0.0),
                    "bytes_tx_MB": gw_metrics.get('bytes_tx_MB', 0.0),
                    "drops_rx_total": gw_metrics.get('drops_rx_total', 0),
                    "drops_tx_total": gw_metrics.get('drops_tx_total', 0),
                    
                    # 🎯 Nouvelles métriques inter-étapes détaillées
                    "interstage_rx_to_cpu_gpu_ms": gw_metrics.get('interstage_rx_to_cpu_gpu_ms', 0.0),
                    "interstage_cpu_gpu_to_proc_ms": gw_metrics.get('interstage_cpu_gpu_to_proc_ms', 0.0),
                    "interstage_proc_to_gpu_cpu_ms": gw_metrics.get('interstage_proc_to_gpu_cpu_ms', 0.0),
                    "interstage_gpu_cpu_to_tx_ms": gw_metrics.get('interstage_gpu_cpu_to_tx_ms', 0.0),
                    "interstage_rx_to_cpu_gpu_p95_ms": gw_metrics.get('interstage_rx_to_cpu_gpu_p95_ms', 0.0),
                    "interstage_cpu_gpu_to_proc_p95_ms": gw_metrics.get('interstage_cpu_gpu_to_proc_p95_ms', 0.0),
                    "interstage_proc_to_gpu_cpu_p95_ms": gw_metrics.get('interstage_proc_to_gpu_cpu_p95_ms', 0.0),
                    "interstage_gpu_cpu_to_tx_p95_ms": gw_metrics.get('interstage_gpu_cpu_to_tx_p95_ms', 0.0),
                    "interstage_samples": gw_metrics.get('interstage_samples', 0),
                })
                LOG.debug("Fused gateway metrics into aggregated: inter-stage samples=%d", 
                         gw_metrics.get('interstage_samples', 0))
        except Exception as e:
            LOG.debug("Failed to collect gateway metrics: %s", e)
    
    return avg  # renvoie les valeurs moyennes (simulées ou fusionnées avec gateway)


def get_gpu_utilization() -> float:
    """Retourne le pourcentage d’utilisation du GPU si disponible, sinon 0.0."""
    try:
        import pynvml  # bibliothèque NVIDIA pour accéder aux informations GPU

        pynvml.nvmlInit()  # initialise la bibliothèque NVML
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # récupère le premier GPU (index 0)
        util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)  # lit le taux d’utilisation GPU
        try:
            pynvml.nvmlShutdown()  # tente de fermer proprement la session NVML
        except Exception:
            pass  # ignore les erreurs à la fermeture
        return util  # retourne le pourcentage d’utilisation GPU
    except Exception:
        return 0.0  # en cas d’erreur (absence de GPU ou NVML), retourne 0.0


def collect_gateway_metrics(gw) -> Dict[str, float]:
    """Collecte les métriques dynamiques à partir d’un objet de type IGTGateway.

    Retourne un dictionnaire prêt à être intégré dans une ligne de KPI.
    Si `gw` est None (aucun gateway actif), retourne un dictionnaire vide.
    """
    if gw is None:  # si aucun objet gateway n’est fourni
        return {}  # retourne un dictionnaire vide
    try:
        snap = gw.stats.snapshot()  # tente de récupérer un instantané des statistiques depuis le gateway
    except Exception:
        snap = {}  # si l’appel échoue, on utilise un dictionnaire vide

    # inclut des champs supplémentaires utiles si présents dans le snapshot
    return {
        "fps_rx": float(snap.get("avg_fps_rx", 0.0)),  # fréquence moyenne des frames reçues (RX)
        "fps_tx": float(snap.get("avg_fps_tx", 0.0)),  # fréquence moyenne des frames envoyées (TX)
        "bytes_rx_MB": float(snap.get("bytes_rx", 0)) / 1e6,  # octets reçus convertis en mégaoctets
        "bytes_tx_MB": float(snap.get("bytes_tx", 0)) / 1e6,  # octets envoyés convertis en mégaoctets
        # le gateway peut exposer une latence moyenne; sinon, on récupère la dernière mesurée
        "latency_ms": float(snap.get("avg_latency_ms", getattr(gw, "last_latency_ms", 0.0) or 0.0)),
        "avg_latency_ms": float(snap.get("avg_latency_ms", getattr(gw, "last_latency_ms", 0.0) or 0.0)),
        # compteurs de pertes (drops) si disponibles
        "drops_rx_total": int(snap.get("drops_rx_total", snap.get("drops_rx", 0) or 0)),
        "drops_tx_total": int(snap.get("drops_tx_total", snap.get("drops_tx", 0) or 0)),
        # fréquence cible configurée dans le gateway
        "fps_target": float(getattr(getattr(gw, "config", None), "target_fps", 0.0) or 0.0),
        
        # 🎯 NOUVELLES MÉTRIQUES INTER-ÉTAPES DÉTAILLÉES (Workflow GPU-résident)
        # Latences moyennes par étape
        "interstage_rx_to_cpu_gpu_ms": float(snap.get("interstage_rx_to_cpu_gpu_ms", 0.0)),
        "interstage_cpu_gpu_to_proc_ms": float(snap.get("interstage_cpu_gpu_to_proc_ms", 0.0)),
        "interstage_proc_to_gpu_cpu_ms": float(snap.get("interstage_proc_to_gpu_cpu_ms", 0.0)),
        "interstage_gpu_cpu_to_tx_ms": float(snap.get("interstage_gpu_cpu_to_tx_ms", 0.0)),
        
        # Percentiles P95 par étape
        "interstage_rx_to_cpu_gpu_p95_ms": float(snap.get("interstage_rx_to_cpu_gpu_p95_ms", 0.0)),
        "interstage_cpu_gpu_to_proc_p95_ms": float(snap.get("interstage_cpu_gpu_to_proc_p95_ms", 0.0)),
        "interstage_proc_to_gpu_cpu_p95_ms": float(snap.get("interstage_proc_to_gpu_cpu_p95_ms", 0.0)),
        "interstage_gpu_cpu_to_tx_p95_ms": float(snap.get("interstage_gpu_cpu_to_tx_p95_ms", 0.0)),
        
        # Métadonnées inter-étapes
        "interstage_samples": int(snap.get("interstage_samples", 0)),
    }


def log_kpi_tick(fps_in: float, fps_out: float, latency_ms: float, gpu_util: float = 0.0) -> None:
    """Écrit une ligne de KPI analysable dans le logger dédié aux KPI."""
    q_metrics = collect_queue_metrics()  # collecte les statistiques courantes sur les queues
    # compose un message compact au format clé=valeur
    try:
        from core.monitoring.kpi import format_kpi, safe_log_kpi  # import local des utilitaires KPI

        kmsg = format_kpi({  # création d’un message structuré avec les mesures principales
            "ts": time.time(),                                  # horodatage en secondes (époque)
            "fps_in": f"{fps_in:.2f}",                          # fréquence d’entrée
            "fps_out": f"{fps_out:.2f}",                        # fréquence de sortie
            "latency_ms": f"{latency_ms:.1f}",                  # latence en millisecondes
            "gpu_util": f"{gpu_util:.1f}",                      # utilisation GPU
            "q_rt": q_metrics['Queue_RT_dyn']['size'],          # taille de la queue temps réel
            "q_gpu": q_metrics['Queue_GPU']['size'],            # taille de la queue GPU
            "drops_rt": q_metrics['Queue_RT_dyn']['drops'],     # nombre de pertes RT
            "drops_gpu": q_metrics['Queue_GPU']['drops'],       # nombre de pertes GPU
        })
        if is_kpi_enabled():  # si la journalisation des KPI est activée
            safe_log_kpi(kmsg)  # écrit la ligne dans logs/kpi.log (via le logger igt.kpi)
    except Exception:
        # en cas d’erreur du module KPI, tentative de repli pour émettre un message structuré
        try:
            from core.monitoring.kpi import format_kpi, safe_log_kpi
            kmsg = format_kpi({
                "ts": time.time(),
                "fps_in": f"{fps_in:.2f}",
                "fps_out": f"{fps_out:.2f}",
                "latency_ms": f"{latency_ms:.1f}",
                "gpu_util": f"{gpu_util:.1f}",
                "q_rt": q_metrics['Queue_RT_dyn']['size'],
                "q_gpu": q_metrics['Queue_GPU']['size'],
                "drops_rt": q_metrics['Queue_RT_dyn']['drops'],
                "drops_gpu": q_metrics['Queue_GPU']['drops'],
            })
            if is_kpi_enabled():
                safe_log_kpi(kmsg)
        except Exception:
            LOG.debug("KPI fallback emission failed; skipping")  # journalise un message si l’émission échoue totalement


def export_kpi_snapshot(filepath: str = "logs/kpi_snapshot.json") -> None:
    """Sauvegarde le dernier instantané de métriques agrégées dans un fichier JSON.

    Crée les répertoires parents si nécessaire.
    """
    try:
        avg = aggregate_metrics(0.0, 0.0, 0.0, 0.0)  # tente d’agréger des métriques par défaut (peut retourner None)
        # si aggregate_metrics retourne None (aucune donnée encore disponible), on crée un snapshot vide
        if avg is None:
            avg = {"fps_in": 0.0, "fps_out": 0.0, "latency_ms": 0.0, "gpu_util": 0.0}
        p = Path(filepath)  # convertit le chemin en objet Path
        p.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier logs/ s’il n’existe pas
        # écrit un fichier JSON horodaté et met à jour la version la plus récente
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())  # format d’horodatage lisible (UTC)
        stamped = p.parent / f"kpi_snapshot_{ts}.json"       # nom de fichier unique par date
        stamped.write_text(json.dumps(avg, indent=2))         # écrit le fichier horodaté
        # met à jour le fichier principal (dernier snapshot courant)
        p.write_text(json.dumps(avg, indent=2))
        # effectue la rotation et compression des anciens fichiers de snapshot
        try:
            rotate_snapshots(p.parent, keep=10)  # conserve les 10 plus récents
        except Exception:
            pass  # ignore les erreurs de rotation
    except Exception:
        LOG.exception("Failed to export KPI snapshot")  # journalise une erreur si l’export échoue


def rotate_snapshots(base_dir: str = "logs", keep: int = 10):
    """Effectue la rotation des fichiers de snapshots KPI dans `base_dir`, en conservant les `keep` plus récents.

    Les fichiers plus anciens que ce seuil sont compressés au format `.json.gz`
    afin de préserver l’historique tout en économisant de l’espace disque.
    Le fichier principal `kpi_snapshot.json` est toujours réécrit avec la dernière version,
    sans lien symbolique, pour garantir la compatibilité entre plateformes.
    """
    base = Path(base_dir)  # crée un objet Path représentant le dossier de base
    if not base.exists():  # si le dossier n’existe pas, on ne fait rien
        return
    # recherche tous les fichiers "kpi_snapshot_*.json" triés du plus récent au plus ancien
    files = sorted(base.glob("kpi_snapshot_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    # garde les `keep` fichiers les plus récents, compresse les autres
    for old in files[keep:]:
        try:
            gz = old.with_suffix(old.suffix + ".gz")  # crée le nom du fichier compressé
            # si la version compressée existe déjà, on supprime simplement l’ancienne
            if gz.exists():
                try:
                    old.unlink()  # supprime l’ancien fichier non compressé
                except Exception:
                    pass
                continue
            import gzip  # module standard Python pour la compression Gzip

            # ouverture du fichier source en lecture binaire et création du fichier compressé
            with old.open("rb") as fh_in, gzip.open(str(gz), "wb") as fh_out:
                fh_out.writelines(fh_in)  # écrit le contenu compressé
            # si la compression réussit, on supprime le fichier original
            try:
                old.unlink()
            except Exception:
                pass
        except Exception:
            # en cas d’échec de compression, on tente de supprimer le fichier pour limiter la taille disque
            try:
                old.unlink()
            except Exception:
                pass


def start_monitor_thread(config: Optional[Dict] = None) -> None:
    """Démarre un thread périodique qui collecte et journalise les métriques de la pipeline.

    Args:
        config: dictionnaire optionnel contenant les paramètres (intervalle, files à surveiller, niveau de log, etc.)
    """
    interval = (config or {}).get("interval_sec", 1.0)  # intervalle de collecte par défaut (1 seconde)
    LOG.info("Monitor thread started (interval=%.2fs)", interval)  # message de démarrage

    def _loop():  # fonction interne exécutée dans un thread indépendant

        while True: # boucle légère : émet des valeurs fictives par défaut sauf si remplacée par des hooks réels
            try:
                # ⚠️ TODO Remplacer les valeurs simulées par de vrais compteurs de la pipeline
                fps_in, fps_out, latency_ms = 25.0, 25.0, 40.0 # Ces valeurs sont simulées ici ; elles seront remplacées plus tard par de vrais compteurs
                gpu_util = get_gpu_utilization() # essaie d’obtenir l’utilisation réelle du GPU (si pynvml est disponible)
                
                gw = (config or {}).get("gateway") if isinstance(config, dict) else None # collecte les métriques du gateway s’il a été passé dans la configuration
                gw_metrics = collect_gateway_metrics(gw) if gw is not None else {}  
                agg = aggregate_metrics(fps_in, fps_out, latency_ms, gpu_util) # enregistre cet échantillon dans l’historique mémoire pour lisser les valeurs
                
                q_metrics = collect_queue_metrics() # compose un message de monitoring contenant les valeurs instantanées et les métriques de queue
                
                msg = (  # construit le message de monitoring complet (toutes les métriques clés)
                    f"[monitor] ts={time.time():.3f} "  # horodatage courant (secondes depuis epoch, arrondi à 3 décimales)
                    f"fps_in={fps_in:.2f} fps_out={fps_out:.2f} "  # fréquences d'entrée/sortie de la pipeline (images/s)
                    f"latency_ms={latency_ms:.1f} gpu_util={gpu_util:.1f} "  # latence moyenne (ms) et taux d’utilisation GPU (%)
                    f"q_rt={q_metrics.get('Queue_RT_dyn', {}).get('size', 0)} "  # taille actuelle de la queue temps réel (nombre d’éléments)
                    f"drops_rt={q_metrics.get('Queue_RT_dyn', {}).get('drops', 0)} "  # nombre d’images perdues dans la queue temps réel
                )


                if gw_metrics: # ajoute les champs relatifs au gateway si présents
                    msg += (  # ajoute au message principal les métriques spécifiques au gateway (réseau IGTLink)
                        f" fps_rx={gw_metrics.get('fps_rx', 0.0):.1f} "  # fréquence de réception depuis PlusServer (images/s)
                        f"fps_tx={gw_metrics.get('fps_tx', 0.0):.1f} "  # fréquence d’émission vers Slicer ou autres clients (images/s)
                        f"MB_rx={gw_metrics.get('bytes_rx_MB', 0.0):.3f} "  # débit de réception (mégaoctets/s)
                        f"MB_tx={gw_metrics.get('bytes_tx_MB', 0.0):.3f} "  # débit d’émission (mégaoctets/s)
                        f"latency_gw={gw_metrics.get('latency_ms', 0.0):.1f} "  # latence moyenne mesurée au niveau du gateway (ms)
                        f"fps_target={gw_metrics.get('fps_target', 0.0):.1f}"  # fréquence cible configurée pour la capture (images/s)
                    )


                if agg is not None: # si une moyenne lissée est disponible, on l’utilise pour remplacer les valeurs brutes
                    try:
                        msg = (  # reconstruit un message de monitoring à partir des moyennes glissantes (valeurs lissées)
                            f"[monitor] ts={time.time():.3f} "  # horodatage courant (en secondes, 3 décimales)
                            f"fps_in={agg['fps_in']:.2f} fps_out={agg['fps_out']:.2f} "  # fréquences d’entrée/sortie moyennes (images/s)
                            f"latency_ms={agg['latency_ms']:.1f} gpu_util={agg['gpu_util']:.1f} "  # latence moyenne et taux d’utilisation GPU (%)
                        ) + (  # ajoute la fin du message précédent contenant les métriques de queue si elles existaient
                            msg.split('q_rt=')[-1] if 'q_rt=' in msg else ''  # préserve les métriques de queue et drops si déjà présentes
                        )
                    except Exception:
                        LOG.debug("Échec du formatage de la moyenne lissée ; utilisation des valeurs instantanées")
                
                if is_kpi_enabled(): # si le système KPI est activé, on logge les métriques sous forme structurée
                    try:
                        from core.monitoring.kpi import format_kpi
                        kdata = {  # dictionnaire complet contenant les données KPI (Key Performance Indicators)
                            "ts": time.time(),  # horodatage actuel en secondes (timestamp UNIX)
                            "fps_in": fps_in,  # fréquence d’arrivée des images dans la pipeline (images/s)
                            "fps_out": fps_out,  # fréquence de sortie des images traitées (images/s)
                            "latency_ms": latency_ms,  # latence moyenne observée entre réception et sortie (millisecondes)
                            "gpu_util": gpu_util,  # taux d’utilisation GPU actuel (%)

                            # informations sur les files internes (queues)
                            "q_rt": q_metrics.get('Queue_RT_dyn', {}).get('size', 0),  # taille de la queue temps réel (nombre d’éléments)
                            "q_gpu": q_metrics.get('Queue_GPU', {}).get('size', 0),  # taille de la queue GPU (frames en attente d’inférence)
                            "drops_rt": q_metrics.get('Queue_RT_dyn', {}).get('drops', 0),  # nombre d’images perdues dans la queue temps réel
                            "drops_gpu": q_metrics.get('Queue_GPU', {}).get('drops', 0),  # nombre d’images perdues dans la queue GPU
                        }

                        if gw_metrics:  # ajout des métriques du gateway si disponibles (réseau IGTLink actif)
                            kdata.update({  # fusionne les métriques du gateway dans le dictionnaire KPI global
                                "fps_rx": gw_metrics.get('fps_rx', 0.0),  # fréquence de réception d’images depuis PlusServer (images/s)
                                "fps_tx": gw_metrics.get('fps_tx', 0.0),  # fréquence d’envoi des images traitées vers Slicer ou autres clients (images/s)
                                "MB_rx": gw_metrics.get('bytes_rx_MB', 0.0),  # débit de réception réseau (mégaoctets/s)
                                "MB_tx": gw_metrics.get('bytes_tx_MB', 0.0),  # débit d’émission réseau (mégaoctets/s)
                                "avg_latency_ms": gw_metrics.get('avg_latency_ms', gw_metrics.get('latency_ms', 0.0)),  # latence moyenne mesurée au niveau du gateway (ms)
                                "drops_rx_total": gw_metrics.get('drops_rx_total', 0),  # nombre total de paquets ou images perdues en réception
                                "drops_tx_total": gw_metrics.get('drops_tx_total', 0),  # nombre total de paquets ou images perdues en émission
                            })
                        kmsg = format_kpi(kdata)  # formate les données KPI en texte clé=valeur
                        safe_log_kpi(kmsg)  # envoie le message vers logs/kpi.log via le logger igt.kpi
                    except Exception:
                        LOG.exception("safe_log_kpi a échoué (monitor)")  # journalise une erreur si l’émission KPI échoue
                else:
                    LOG.info("KPI: %s", msg)  # sinon logge le message textuel dans le logger général
                
                try: # export périodique d’un snapshot JSON (tous les N ticks, configurable)
                    tick = getattr(start_monitor_thread, "_tick_counter", 0) + 1  # incrémente le compteur interne
                    setattr(start_monitor_thread, "_tick_counter", tick)  # stocke le compteur interne de ticks dans la fonction (sert à suivre le nombre d’itérations du thread)
                    export_every = (config or {}).get("snapshot_every", 10) if isinstance(config, dict) else 10  # définit la fréquence d’export ou de snapshot des métriques (par défaut toutes les 10 itérations)
                    if tick % export_every == 0:  # tous les N intervalles, on exporte un snapshot
                        export_kpi_snapshot("logs/kpi_snapshot.json")
                except Exception:
                    LOG.debug("export_kpi_snapshot ignoré (erreur mineure)")
                time.sleep(interval)  # attend la durée spécifiée avant la prochaine itération
            except Exception:
                LOG.exception("Exception dans la boucle du thread de monitoring")  # capture toute erreur pour éviter l’arrêt du thread

    import threading
    # ⚠️ TODO S’exécute indéfiniment tant que le programme tourne, Pas de stop_event → le thread ne peut pas être arrêté proprement pour l’instant.
    threading.Thread(target=_loop, name="MonitorThread", daemon=True).start()  # démarre le thread en mode démon
    # note : implémentation complète (join, stop event, etc.) non encore présente
    return


def log_pipeline_metrics(
    fps_in: float,
    fps_out: float,
    latency_ms: float,
    gpu_util: float,
    extras: Optional[Dict] = None
) -> None:
    """Journalise les métriques principales de la pipeline.

    Args:
        fps_in: nombre d'images par seconde en entrée (avant traitement)
        fps_out: nombre d'images par seconde en sortie (après traitement)
        latency_ms: latence totale de la pipeline en millisecondes
        gpu_util: taux d'utilisation du GPU en pourcentage
        extras: dictionnaire optionnel contenant des métriques additionnelles personnalisées
    """
    # Écrit un message d'information standard dans le logger principal
    LOG.info(
        "Pipeline metrics fps_in=%s fps_out=%s latency_ms=%s gpu_util=%s",
        fps_in, fps_out, latency_ms, gpu_util
    )
    # Si des métriques supplémentaires sont présentes et que le niveau DEBUG est activé, on les loggue
    if extras and LOG.isEnabledFor(logging.DEBUG):
        LOG.debug("Extra metrics: %s", extras)
    return  # rien à renvoyer (simple enregistrement de logs)


def adjust_rt_queue_size(current_size: int, metrics: Dict) -> int:
    # ⚠️ TODO fonction utilitaire prévue pour ajuster dynamiquement la taille de la file temps réel principale (Queue_RT_dyn)
    # future implémentation PID
    """Calcule et retourne la nouvelle taille souhaitée pour la file (queue) temps réel dynamique.

    Args:
        current_size: taille actuelle de la queue
        metrics: dictionnaire de métriques récentes (ex : tailles des queues, fps, latence)

    Returns:
        La nouvelle taille (int), ajustée selon les métriques. Ici, retourne la taille actuelle (fonction factice).
    """
    # ⚠️ TODO Fonction encore non implémentée : pour l’instant, renvoie simplement la taille actuelle.
    # future implémentation PID
    LOG.debug("adjust_rt_queue_size appelé avec current_size=%s", current_size)
    return current_size  # aucune modification de la taille n’est encore appliquée


def get_pipeline_metrics() -> Dict:
    """Retourne un instantané des métriques courantes de la pipeline, incluant les KPI des files et un horodatage.

    Structure du retour :
        {
            "queues": { ... },   # dictionnaire des métriques par queue (taille, drops, etc.)
            "timestamp": <float> # horodatage UNIX (secondes)
        }
    """
    # collect_queue_metrics() retourne l’état actuel de chaque file : taille, nombre de drops, etc.
    return {"queues": collect_queue_metrics(), "timestamp": time.time()}
