"""Logique du thread de supervision, séparée du gestionnaire principal (gateway manager).

Ce module implémente un petit contrôleur qui lit périodiquement les statistiques 
du gateway et émet des messages KPI via core.monitoring.kpi.safe_log_kpi si disponible, 
sinon utilise un fallback vers le logging standard.
"""
import time  # gestion du temps et des temporisations
import logging  # module standard pour le journal des événements
from typing import Optional, Any, Dict  # types pour annotations de paramètres optionnels et génériques
from service.heartbeat import measure_latency  # fonction utilitaire pour mesurer la latence réseau
from service.autotune import AutoTuner  # module d’ajustement automatique des paramètres

LOG = logging.getLogger("igt.gateway.supervisor")  # création d’un logger spécifique pour le superviseur


class SupervisorThread:
    def __init__(self, stats, stop_event, interval_s: float = 2.0,
                 threads: Optional[Dict[str, Any]] = None, events: Optional[Any] = None,
                 buffers: Optional[Dict[str, Any]] = None, config: Optional[Any] = None) -> None:
        self.stats = stats  # référence vers l’objet GatewayStats pour accéder aux métriques
        self.stop_event = stop_event  # événement global utilisé pour signaler l’arrêt du thread
        self.interval_s = interval_s  # intervalle en secondes entre deux cycles de supervision
        self.threads = threads or {}  # dictionnaire des threads à surveiller (ex: {"rx": thread_RX, "tx": thread_TX})
        self.events = events  # gestionnaire d’événements partagé (EventEmitter)
        self.buffers = buffers or {}  # dictionnaire des files internes surveillées (mailbox, outbox, etc.)
        self.config = config  # configuration globale du système (adresses, FPS, etc.)
        self._autotuner = AutoTuner(config) if config is not None else None  # initialisation optionnelle du module d’auto-tuning

        # safe_log_kpi est optionnel ; on l’importe seulement si le module de monitoring est disponible
        try:
            from core.monitoring.kpi import safe_log_kpi  # import paresseux de la fonction d’enregistrement des KPI
        except Exception:
            safe_log_kpi = None  # si le module n’existe pas, on désactive la fonction KPI
            LOG.debug("core.monitoring.kpi.safe_log_kpi non disponible ; utilisation du logging par défaut")  # message de debug si fallback activé

        self._safe_log_kpi = safe_log_kpi  # conserve la référence vers la fonction KPI (ou None si absente)

    def run(self) -> None:  
        LOG.info("Supervisor thread started (interval=%.2fs)", self.interval_s)  # log initial indiquant le démarrage du superviseur avec son intervalle de cycle
        while not self.stop_event.is_set():  # boucle principale : tourne tant que le signal d’arrêt n’a pas été activé
            try:
                snapshot = self.stats.snapshot()  # capture instantanée des statistiques courantes depuis GatewayStats
                now = time.time()  # enregistre l’heure actuelle (timestamp système)
                fps_rx = snapshot.get("avg_fps_rx", 0.0)  # récupère le FPS moyen du flux de réception (RX)
                fps_tx = snapshot.get("avg_fps_tx", 0.0)  # récupère le FPS moyen du flux d’envoi (TX)

                # appliquer une légère décroissance (decay) si les données ne sont plus récentes
                last_rx = snapshot.get("last_update_rx", 0.0)  # timestamp de la dernière mise à jour RX
                last_tx = snapshot.get("last_update_tx", 0.0)  # timestamp de la dernière mise à jour TX
                if now - last_rx > 5.0:  # si plus de 5 secondes se sont écoulées depuis la dernière frame reçue
                    fps_rx = max(0.0, fps_rx * 0.5)  # réduit artificiellement le FPS RX pour refléter une perte d’activité
                if now - last_tx > 5.0:  # si plus de 5 secondes sans nouvelle transmission TX
                    fps_tx = max(0.0, fps_tx * 0.5)  # réduit le FPS TX de moitié également

                mb_rx = snapshot.get("bytes_rx", 0) / 1e6  # convertit le total d’octets reçus en mégaoctets
                mb_tx = snapshot.get("bytes_tx", 0) / 1e6  # convertit le total d’octets transmis en mégaoctets

                msg = (  # formatte un message synthétique contenant les indicateurs réseau
                    f"[supervisor] net ts={now:.3f} fps_rx={fps_rx:.1f} fps_tx={fps_tx:.1f} "
                    f"MB_rx={mb_rx:.3f} MB_tx={mb_tx:.3f}"
                )

                if self._safe_log_kpi:  # si la fonction de log KPI (core.monitoring.kpi.safe_log_kpi) est disponible
                    try:
                        from core.monitoring.kpi import format_kpi

                        kmsg = format_kpi({"ts": now, "event": "net", "fps_rx": f"{fps_rx:.1f}", "fps_tx": f"{fps_tx:.1f}", "MB_rx": f"{mb_rx:.3f}", "MB_tx": f"{mb_tx:.3f}"})
                        self._safe_log_kpi(kmsg)  # enregistre le message KPI via le module de monitoring
                    except Exception:
                        LOG.exception("safe_log_kpi failed")  # log d’erreur si l’écriture KPI échoue
                else:
                    LOG.info("KPI: %s", msg)  # sinon, fallback vers un simple log standard INFO

                # --- Rééquilibrage dynamique des files (fonction expérimentale) ---
                try:
                    from core.queues.adaptive import adjust_queue_size  # importe la fonction d’ajustement automatique des tailles de queue
                                                                         
                    for name, q in list(self.buffers.items()):  # parcourt chaque file (mailbox, outbox, etc.) surveillée par le superviseur
                        try:
                            new_q, new_len = adjust_queue_size(q, fps_rx, fps_tx, mb_rx, mb_tx)  # appelle la fonction pour ajuster dynamiquement la taille selon l’activité RX/TX
                            
                            # si la fonction retourne un nouvel objet (ex. remplacement complet du deque), on met à jour la référence
                            if new_q is not q:
                                self.buffers[name] = new_q  # met à jour le dictionnaire interne pour pointer vers la nouvelle file
                            
                            # formate un message KPI indiquant la taille actuelle de la queue
                            if self._safe_log_kpi:  # si le logger KPI est disponible
                                try:
                                    from core.monitoring.kpi import format_kpi

                                    kmsg = format_kpi({"ts": time.time(), "event": "queue_size", "name": name, "size": new_len})
                                    self._safe_log_kpi(kmsg)  # envoie la taille de la queue au système KPI
                                except Exception:
                                    LOG.debug("safe_log_kpi failed (queue size)")  # message de debug si l’écriture KPI échoue
                            else:
                                LOG.info("KPI: %s", kmsg)  # fallback : enregistre la taille via le logger standard
                        except Exception:
                            LOG.debug(f"{name} queue resize skipped")  # si une erreur survient sur une file, on la passe et continue la boucle
                except Exception:
                    LOG.exception("Dynamic queue resizing failed")  # capture toute exception globale sur le bloc d’ajustement dynamique

                # --- Watchdog : vérifie que les threads principaux sont toujours actifs ---
                try:
                    for name, thread in list(self.threads.items()):  # parcourt chaque thread surveillé (rx, tx, etc.)
                        try:
                            alive = thread.is_alive() if thread is not None else False  # teste si le thread est toujours vivant
                        except Exception:
                            alive = False  # en cas d’erreur d’accès, considère le thread comme inactif
                        if not alive:  # si le thread ne tourne plus
                            LOG.warning("Thread %s is dead (detected by watchdog)", name)  # avertissement dans les logs
                            if self.events:  # si un gestionnaire d’événements est disponible
                                try:
                                    # émet un événement indiquant la mort du thread, avec son nom et un horodatage
                                    self.events.emit("thread_dead", {"name": name, "ts": time.time()})
                                except Exception:
                                    LOG.exception("Failed to emit thread_dead event")  # log d’erreur si l’émission échoue
                except Exception:
                    LOG.exception("Exception in SupervisorThread loop (watchdog)")  # capture toute erreur du bloc watchdog

                # --- Heartbeat : mesure périodique de la latence réseau ---
                try:
                    if not hasattr(self, "_hb_counter"):  # vérifie si le compteur interne d’itérations existe
                        self._hb_counter = 0  # initialise le compteur s’il n’existe pas encore
                    self._hb_counter += 1  # incrémente le compteur d’itérations

                    # mesure toutes les 10 itérations (≈ toutes les 10 × interval_s secondes)
                    if self._hb_counter % 10 == 0:
                        # TODO : rendre host/port configurables ; pour l’instant, utilise localhost:18944 (PlusServer)
                        latency = measure_latency("127.0.0.1", 18944, timeout=1.0)  # mesure la latence réseau en millisecondes
                        if latency >= 0.0:  # si la mesure a réussi (valeur valide)
                            msg_latency = f"[supervisor] latency_ms={latency:.1f}"  # formate un message KPI pour la latence
                            # conserve la dernière valeur mesurée pour l’auto-tuner
                            try:
                                self._last_latency_ms = latency  # stocke la latence pour ajustement adaptatif ultérieur
                            except Exception:
                                pass  # ignore silencieusement si l’attribut ne peut être mis à jour
                            if self._safe_log_kpi:  # si le module KPI est disponible
                                try:
                                    from core.monitoring.kpi import format_kpi

                                    kmsg = format_kpi({"ts": time.time(), "event": "latency", "latency_ms": f"{latency:.1f}"})
                                    self._safe_log_kpi(kmsg)  # envoie la mesure de latence au système KPI
                                except Exception:
                                    LOG.exception("safe_log_kpi failed (latency)")  # erreur si l’écriture KPI échoue
                            else:
                                LOG.info("KPI: %s", msg_latency)  # sinon log standard avec la latence mesurée
                        else:
                            LOG.warning("Latency heartbeat failed (unreachable)")  # avertissement si la mesure échoue (hôte injoignable)
                except Exception:
                    LOG.exception("Heartbeat measurement failed")  # capture toute exception dans le bloc de mesure de latence

                # --- Auto-tuning : adaptation automatique du FPS cible et de l’intervalle de supervision ---
                try:
                    if not hasattr(self, "_tune_counter"):  # vérifie si le compteur d’auto-tuning existe déjà
                        self._tune_counter = 0  # initialise le compteur si inexistant
                    self._tune_counter += 1  # incrémente le compteur à chaque cycle

                    # tous les 10 cycles (≈ toutes les 10 × interval_s secondes) et si un AutoTuner est actif
                    if self._tune_counter % 10 == 0 and self._autotuner:
                        avg_latency = getattr(self, "_last_latency_ms", 0.0)  # récupère la dernière latence moyenne mesurée (0.0 par défaut)

                        # sélectionne la taille maximale de queue observée comme indicateur de pression du système
                        qlen = 0
                        try:
                            qlen = max((getattr(q, "maxlen", 0) or 0) for q in self.buffers.values())  # lit maxlen pour chaque buffer (mailbox/outbox)
                        except Exception:
                            qlen = 0  # si une erreur survient, valeur par défaut 0

                        self._autotuner.tune(avg_latency, qlen)  # appelle l’AutoTuner pour ajuster les paramètres (fps cible, intervalle supervision)

                        try:
                            # formate un message KPI résumant les nouveaux paramètres ajustés
                            if self._safe_log_kpi:
                                from core.monitoring.kpi import format_kpi

                                kmsg = format_kpi({"ts": time.time(), "event": "autotune", "fps_target": f"{self.config.target_fps:.1f}", "interval": f"{self.config.supervise_interval_s:.1f}", "latency_ms": f"{avg_latency:.1f}"})
                                self._safe_log_kpi(kmsg)
                            else:
                                msg_tune = f"[autotune] fps_target={self.config.target_fps:.1f} interval={self.config.supervise_interval_s:.1f} latency_ms={avg_latency:.1f}"
                                LOG.info("KPI: %s", msg_tune)  # sinon log standard
                        except Exception:
                            LOG.exception("Failed to emit auto_tune KPI")  # capture toute erreur de log KPI
                except Exception:
                    LOG.exception("Auto-tuning failed")  # capture toute erreur du bloc d’adaptation automatique

                # attend l’intervalle défini ou une demande d’arrêt avant de relancer un cycle de supervision
                self.stop_event.wait(self.interval_s)  # temporisation passive avec sortie anticipée si stop_event est activé
            except Exception:
                LOG.exception("Exception in SupervisorThread loop")  # capture toute erreur inattendue survenue dans la boucle principale

        LOG.info("Supervisor thread stopped cleanly.")  # message final confirmant l’arrêt propre du thread de supervision
