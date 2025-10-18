"""High-level gateway manager (public IGTGateway class).

This class focuses on orchestration only and delegates stats, events and
supervision to dedicated components.
"""
from __future__ import annotations

# === Standard Library ===
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Union

# === Project Modules ===
from service.registry import THREAD_REGISTRY
from core.types import RawFrame
from core.queues.adaptive import AdaptiveDeque
from service.gateway.config import GatewayConfig
from service.gateway.stats import GatewayStats
from service.gateway.events import EventEmitter
from service.gateway.supervisor import SupervisorThread

# === Logger ===
LOG = logging.getLogger("igt.service")

class IGTGateway:
    """
    Orchestrateur OpenIGTLink principal.
    
    Rôle :
      - coordonner les threads RX (réception), TX (envoi) et Supervisor (monitoring),
      - gérer les files internes de frames et de masques (`AdaptiveDeque`),
      - exposer les statistiques et événements système.
    
    Cette classe ne traite pas directement les images ; elle contrôle le flux
    et la stabilité de la passerelle IGTLink en s’appuyant sur les composants
    spécialisés (`GatewayStats`, `SupervisorThread`, etc.).
    """

    def __init__(
        self,
        config_or_host: Union[GatewayConfig, str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le gestionnaire de passerelle IGTLink.
        
        Deux modes de construction :
            1 Recommandé : avec un objet GatewayConfig pré-initialisé.
              ex: IGTGateway(GatewayConfig.from_yaml("src/config/gateway.yaml"))

            2 Legacy : avec des arguments explicites (pour compatibilité).
              ex: IGTGateway("127.0.0.1", 18944, 18945, target_fps=25.0)
        """
        # --- Résolution et validation de la configuration
        if isinstance(config_or_host, GatewayConfig):
            config: GatewayConfig = config_or_host
        else:
            plus_host: str = str(config_or_host)
            if len(args) < 2:
                raise ValueError(
                    "Legacy constructor requires plus_port and slicer_port as positional args"
                )

            plus_port: int = int(args[0])
            slicer_port: int = int(args[1])
            target_fps: float = float(kwargs.get("target_fps", 25.0))
            supervise_interval_s: float = float(kwargs.get("supervise_interval_s", 2.0))

            config = GatewayConfig(
                plus_host=plus_host,
                plus_port=plus_port,
                slicer_port=slicer_port,
                target_fps=target_fps,
                supervise_interval_s=supervise_interval_s,
            )

        # --- Stockage des paramètres principaux
        self.config: GatewayConfig = config                 # objet dataclass typé chargé depuis YAML (configuration complète de référence, immuable à runtime)
        self.plus_host: str = config.plus_host              # adresse IP / hostname du serveur PlusServer
        self.plus_port: int = config.plus_port              # port TCP d’entrée pour la réception d’images IGTLink
        self.slicer_port: int = config.slicer_port          # port TCP de sortie pour l’envoi des masques vers 3D Slicer
        self.target_fps: float = config.target_fps          # cadence cible (images/s) visée par la passerelle
        self.supervise_interval_s: float = config.supervise_interval_s  # intervalle (s) entre deux cycles de supervision

        # --- Gestion des threads
        self._stop_event: threading.Event = threading.Event()  # signal d’arrêt partagé entre tous les threads (permet un arrêt propre et coordonné)
        self._rx_thread: Optional[threading.Thread] = None     # thread de réception des images et poses (depuis PlusServer)
        self._tx_thread: Optional[threading.Thread] = None     # thread d’envoi des résultats (vers 3D Slicer)
        self._supervisor_thread: Optional[threading.Thread] = None  # thread de supervision (surveille débit, latence, état des threads)
        self._running: bool = False                            # indicateur global d’état du service (True = threads actifs)

        # --- Buffers internes (queues adaptatives)
        self._mailbox: AdaptiveDeque[RawFrame] = AdaptiveDeque(maxlen=2)  # buffer d’entrée à faible latence (RawFrame brutes depuis PlusServer)
        self._outbox: AdaptiveDeque[tuple[Any, Dict[str, Any]]] = AdaptiveDeque(maxlen=8)  # buffer de sortie (masques + métadonnées destinés à 3D Slicer)


        # --- Composants auxiliaires ---
        # autorise une configuration optionnelle des statistiques via la section config.stats
        stats_cfg = getattr(config, "stats", None)  # récupère la sous-section "stats" du fichier de configuration si elle existe, sinon renvoie None (permet de personnaliser la taille des fenêtres de mesure)
        
        if stats_cfg and isinstance(stats_cfg, dict): # Vérifie si la configuration des statistiques est valide
            rolling_size = int(stats_cfg.get("rolling_window_size", 20))  # lit la taille de la fenêtre glissante utilisée pour calculer les moyennes de fps (par défaut 20)
            latency_size = int(stats_cfg.get("latency_window_size", 200))  # lit la taille maximale du buffer de latence RX→TX (par défaut 200 échantillons)
            
            self.stats: GatewayStats = GatewayStats( # crée une instance du collecteur de statistiques GatewayStats
                rolling_window_size=rolling_size, # avec les tailles de fenêtres spécifiées dans la configuratio
                latency_window_size=latency_size  # avec les tailles de fenêtres spécifiées dans la configuration
            )  
            
        else: # si aucune section "stats" n’est définie, instancie GatewayStats avec les valeurs par défaut (20 / 200)
            self.stats: GatewayStats = GatewayStats()  

        self.events: EventEmitter = EventEmitter()  # initialise le gestionnaire d’événements internes, utilisé pour émettre des signaux asynchrones (erreurs, notifications, état de threads, etc.)

        self._last_rx_ts_local: Optional[float] = None  # dernier timestamp local (time.time) — utilisé pour estimer un fps de secours si aucune horodatation RX n’est fournie
        # --- Journalisation d’initialisation
        LOG.info(
            "IGTGateway initialized (plus=%s:%d, slicer_port=%d, fps=%.1f, interval=%.1fs)",
            self.plus_host,
            self.plus_port,
            self.slicer_port,
            self.target_fps,
            self.supervise_interval_s,
        )

    @property
    def buffers(self) -> Dict[str, Any]:
        """Expose les buffers internes de la passerelle pour le superviseur (lecture seule)."""
        return {"mailbox": self._mailbox, "outbox": self._outbox}  # renvoie un dictionnaire contenant les files d’entrée (mailbox) et de sortie (outbox)

    @property
    def is_running(self) -> bool:
        return (self._running and not self._stop_event.is_set())  # retourne True si la passerelle est active et qu’aucun signal d’arrêt n’a été déclenché

    def start(self) -> None:
        if self.is_running:  # empêche de relancer le service s’il est déjà en cours d’exécution
            LOG.warning("IGTGateway déjà en cours d’exécution — démarrage ignoré")
            return

        LOG.info("Démarrage de IGTGateway (PlusServer=%s:%d, Slicer=%d)",
                 self.plus_host, self.plus_port, self.slicer_port)  # message d’information indiquant le démarrage du service avec les ports configurés

        self._stop_event.clear()  # réinitialise le signal d’arrêt partagé (permet un nouveau cycle de threads)
        self._running = True  # marque la passerelle comme active

        LOG.debug("Thread registry chargé : %s", list(THREAD_REGISTRY.keys()))  # affiche la liste des fonctions enregistrées dans le registre global (pour débogage)

        try:
            rx_target = THREAD_REGISTRY["rx"]  # THREAD_REGISTRY est un dictionnaire global contenant une table de correspondance statique entre des noms de threads ("rx", "tx") et leurs fonctions associées ; ici on récupère la fonction liée à la clé "rx" (cible du thread de réception)
        except KeyError:
            raise RuntimeError("THREAD_REGISTRY ne contient pas l’entrée 'rx' (vérifier service.registry)")  # erreur explicite si la fonction RX n’est pas trouvée

        rx_args = (  # arguments passés au thread RX lors de sa création
            self._mailbox,           # file d’entrée pour stocker les frames reçues
            self._stop_event,        # signal d’arrêt partagé
            self.plus_host,          # adresse du serveur PlusServer
            self.plus_port,          # port de réception IGTLink
            self.update_rx_stats,    # fonction de rappel pour mettre à jour les statistiques RX
            self.events.emit,        # émetteur d’événements (notifications, erreurs)
        )

        try:
            tx_target = THREAD_REGISTRY["tx"]  # THREAD_REGISTRY est un dictionnaire global contenant une table de correspondance statique entre des noms de threads ("rx", "tx") et leurs fonctions associées ; ici on récupère la fonction liée à la clé "tx" (cible du thread de transmission vers 3D Slicer)
        except KeyError:
            raise RuntimeError("THREAD_REGISTRY ne contient pas l’entrée 'tx' (vérifier service.registry)")  # erreur explicite si la fonction TX est absente

        tx_args = (  # arguments passés au thread TX lors de sa création
            self._outbox,            # file de sortie contenant les masques et métadonnées à envoyer
            self._stop_event,        # signal d’arrêt partagé
            self.slicer_port,        # port de destination pour 3D Slicer
            self.update_tx_stats,    # fonction de rappel pour mettre à jour les statistiques TX
            self.events.emit,        # émetteur d’événements (notifications, erreurs)
        )

        self._rx_thread = threading.Thread(
            target=rx_target,  # fonction exécutée par le thread (ici run_plus_client → réception depuis PlusServer)
            name="IGT-RX",     # nom du thread (utile pour logs et débogage)
            args=rx_args,      # paramètres passés à la fonction : (mailbox, stop_event, host, port, stats_cb, event_cb)
            daemon=True        # mode démon : le thread s’arrête automatiquement à la fermeture du programme principal
        )  # création du thread RX chargé de recevoir les images IGTLink depuis PlusServer et de les placer dans la mailbox

        self._tx_thread = threading.Thread(target=tx_target, name="IGT-TX", args=tx_args, daemon=True)  # création du thread TX en mode démon (arrêt automatique)

        self._rx_thread.start()  # démarre le thread RX (réception des données depuis PlusServer)
        self._tx_thread.start()  # démarre le thread TX (envoi des résultats vers 3D Slicer)

        sup = SupervisorThread(  # création du superviseur chargé de surveiller les threads et métriques
            stats=self.stats,  # accès au collecteur de statistiques GatewayStats
            stop_event=self._stop_event,  # signal d’arrêt partagé
            interval_s=self.supervise_interval_s,  # fréquence des cycles de supervision (en secondes)
            threads={"rx": self._rx_thread, "tx": self._tx_thread},  # dictionnaire des threads à surveiller
            events=self.events,  # système d’événements partagé
            buffers=self.buffers,  # accès aux buffers internes pour la supervision
            config=self.config,  # configuration complète de la passerelle
        )

        self._supervisor_thread = threading.Thread(target=sup.run, name="IGT-Supervisor", daemon=True)  # création du thread de supervision
        self._supervisor_thread.start()  # démarre le thread de supervision (veille et mesures périodiques)

        LOG.info("IGTGateway démarré avec succès.")  # message de confirmation indiquant que tous les threads sont opérationnels

    def stop(self) -> None:  # Arrête proprement tous les threads du gateway (RX, TX, Supervisor).
        if not self.is_running:  # Vérifie si le gateway est déjà arrêté.
            LOG.warning("IGTGateway not running — stop() ignored.")  # Avertit si l'arrêt est inutile.
            return  # Sort sans rien faire.

        LOG.info("Stopping IGTGateway…")  # Journalise le début de la séquence d'arrêt.
        start_t = time.time()  # Enregistre le temps de départ pour mesurer la durée totale.

        self._stop_event.set()  # Déclenche le signal d'arrêt partagé entre tous les threads.

        for t in (self._rx_thread, self._tx_thread, self._supervisor_thread):  # Parcourt tous les threads à arrêter.
            if t and t.is_alive():  # Vérifie que le thread existe et qu'il est encore actif.
                try:
                    t.join(timeout=1.0)  # Attend jusqu’à 1 seconde que le thread se termine proprement.
                except Exception:
                    LOG.debug("Exception while joining thread %s", getattr(t, "name", t))  # Log en cas d'erreur de join.

        self._running = False  # Met à jour l’état interne pour indiquer que le gateway est arrêté.
        LOG.info("IGTGateway stopped cleanly (%.2fs)", time.time() - start_t)  # Log final avec durée totale d'arrêt.


    def get_status(self) -> Dict[str, Any]:  # Retourne un instantané de l’état courant du gateway.
        snap = self.stats.snapshot()  # Récupère une copie des statistiques courantes (fps, timestamps, etc.).
        return {  # Construit un dictionnaire d’état synthétique pour supervision ou API.
            "client_alive": bool(self._rx_thread and self._rx_thread.is_alive()),  # True si le thread RX (client) est actif.
            "server_alive": bool(self._tx_thread and self._tx_thread.is_alive()),  # True si le thread TX (serveur) est actif.
            "fps_rx": snap.get("fps_rx", 0.0),  # Dernier débit d’images reçues (frames/s).
            "fps_tx": snap.get("fps_tx", 0.0),  # Dernier débit d’images envoyées (frames/s).
            "last_rx_ts": snap.get("last_rx_ts", 0.0),  # Horodatage de la dernière image reçue.
        }


    def receive_image(self) -> Optional[RawFrame]:  # Récupère la dernière image reçue depuis la mailbox (file d’entrée temps réel).
        if not self._mailbox:  # Si la mailbox n’existe pas ou est vide, on interrompt la lecture.
            return None  # Aucun frame à lire, retourne None.

        try:
            frame = self._mailbox.pop()  # Extrait le plus récent RawFrame de la mailbox (FIFO à faible latence).
            # ⚠️ TODO: vérifier/potentiellement adapter la politique de purge (clear) — actuellement drop-oldest pour garantir une fraîcheur temps réel
            self._mailbox.clear()  # Vide la mailbox pour ne conserver qu’une seule frame à traiter (préserve la fraîcheur des données).
            try:
                now = time.time()  # Capture le timestamp système local au moment de la réception (référence de calcul du fps local).
                self.stats.set_last_rx_ts(frame.meta.ts)  # Met à jour l’horodatage RX officiel via l’API de GatewayStats.

                # Si PlusServer fournit déjà un champ fps, on n’a pas besoin de l’estimer localement.
                if getattr(frame.meta, "fps", None) is not None:  # Vérifie si le champ fps existe dans les métadonnées.
                    self._last_rx_ts_local = now  # Met simplement à jour la référence temporelle locale.
                else:
                    # Si aucun fps n’est fourni, on calcule un fps local basé sur le delta de temps entre deux frames successives.
                    if getattr(self, "_last_rx_ts_local", None) is not None:  # Vérifie qu’un timestamp local précédent existe.
                        delta = now - self._last_rx_ts_local  # Calcule le temps écoulé depuis la dernière frame.
                        if delta > 0:  # Évite une division par zéro.
                            try:
                                fps_est = 1.0 / delta  # Estime le fps local (inverse du delta temps).
                                self.stats.update_rx(fps_est, now)  # Met à jour les statistiques RX avec ce fps estimé.
                            except Exception:
                                pass  # Ignore silencieusement les erreurs de calcul ou de mise à jour.
                    self._last_rx_ts_local = now  # Met à jour le timestamp local pour la prochaine itération.

                # Si la frame contient un identifiant, on enregistre son timestamp RX pour mesurer la latence RX→TX plus tard.
                try:
                    fid = getattr(frame.meta, "frame_id", None)  # Récupère l’identifiant de frame s’il existe.
                    if fid is not None:  # Vérifie que l’ID est valide.
                        self.stats.mark_rx(fid, float(getattr(frame.meta, "ts", frame.meta.ts)))  # Enregistre le moment de réception pour le calcul futur de latence.
                except Exception:
                    pass  # Ignore toute erreur liée aux métadonnées incomplètes ou mal formées.

            except Exception:
                pass  # Ignore silencieusement les erreurs si la frame n’a pas de métadonnées ou si les stats ne peuvent pas être mises à jour.

            return frame  # Retourne la frame extraite de la mailbox pour traitement en aval.

        except Exception:
            return None  # En cas d’erreur inattendue (accès concurrent, corruption mémoire, etc.), retourne None.

    def send_mask(self, mask_array: Any, meta: Dict[str, Any]) -> bool:  # Envoie un masque de segmentation et ses métadonnées vers 3D Slicer.
        """
        Enfile (enqueue) un masque et ses métadonnées pour transmission à 3D Slicer.

        Notes :
            - Le dictionnaire `meta` est automatiquement enrichi avec les métriques
            de performance du gateway :
                • latency_ms_avg, latency_ms_p95 → latence moyenne et 95ᵉ percentile
                • snapshot_count / stats_seq → compteur d’échantillons
            - Les clés existantes dans `meta` ne sont pas écrasées.
            - Retourne True si le masque a été ajouté avec succès, False sinon.
        """
        try:
            if len(self._outbox) >= self._outbox.maxlen:  # Si la file de sortie (_outbox) est pleine…
            # ⚠️ TODO [ARCHI]: la suppression du plus ancien masque (drop-oldest) peut casser la correspondance frame_id <-> RawFrame d’origine.
            #    - Actuellement: priorité à la fraîcheur du flux.
            #    - À revoir: implémenter une politique de synchronisation stricte (ex: ne dropper que les masques sans frame correspondante).
            #    - Objectif futur: garantir superposition exacte entre l’image d’entrée et la segmentation affichée dans Slicer.
                try:
                    self._outbox.popleft()  # Supprime le plus ancien masque (politique drop-oldest pour préserver la fraîcheur).
                    from core.monitoring.kpi import increment_drops  # Import différé du compteur de drops (suivi KPI).
                    try:
                        increment_drops("tx.drop_total", 1, emit=True)  # Incrémente la métrique globale "tx.drop_total" de +1 et la publie.
                    except Exception:
                        LOG.debug("increment_drops failed or not available")  # Si la fonction KPI est absente, log de debug seulement.
                except Exception:
                    LOG.debug("Failed to drop oldest from outbox")  # Si la suppression échoue (erreur de file), on ignore.

            # --------------------------------------------------------------
            # Bloc d’injection des métriques de latence :
            # Ajoute les statistiques de performance récentes (moyenne, p95, compteur)
            # dans les métadonnées transmises à 3D Slicer.
            # --------------------------------------------------------------
            try:
                snap = self.stats.snapshot()  # Récupère un instantané des statistiques actuelles du Gateway.
                meta.setdefault("latency_ms_avg", float(snap.get("latency_ms_avg", 0.0)))  # Injecte la latence moyenne (si absente).
                meta.setdefault("latency_ms_p95", float(snap.get("latency_ms_p95", 0.0)))  # Injecte la latence 95ᵉ percentile.
                stats_seq = int(snap.get("snapshot_count", 0))  # Compteur de snapshots (nombre d’itérations statistiques).
                meta.setdefault("stats_seq", stats_seq)  # Clé rétrocompatible avec anciens clients.
                meta.setdefault("snapshot_count", stats_seq)  # Même valeur pour compatibilité montante.

                # Log de confirmation en mode DEBUG
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug(
                        "Injected latency stats into meta: avg=%.2f p95=%.2f seq=%d",
                        meta.get("latency_ms_avg", 0.0),
                        meta.get("latency_ms_p95", 0.0),
                        meta.get("stats_seq", meta.get("snapshot_count", 0)),
                    )
            except Exception:
                pass  # Si l’injection échoue (métadonnées manquantes ou stats indisponibles), on ignore silencieusement.

            self._outbox.append((mask_array, meta))  # Empile le couple (masque, métadonnées) dans la file de sortie.

            # --- Bloc de marquage TX pour calcul de latence RX→TX ---
            try:
                fid = meta.get("frame_id", None)  # Récupère l’identifiant de frame associé (s’il existe).
                tx_ts = meta.get("ts", time.time())  # Timestamp d’envoi (par défaut temps actuel).
                if fid is not None:
                    self.stats.mark_tx(int(fid), float(tx_ts))  # Enregistre le timestamp TX pour cette frame dans GatewayStats.
                    if LOG.isEnabledFor(logging.DEBUG):
                        snap2 = self.stats.snapshot()  # Récupère les stats mises à jour après le marquage TX.
                        instant = float(snap2.get("latency_ms_max", 0.0))  # Dernière latence mesurée (en ms).
                        avg = float(snap2.get("latency_ms_avg", 0.0))  # Moyenne actuelle de latence.
                        p95 = float(snap2.get("latency_ms_p95", 0.0))  # 95ᵉ percentile de latence.
                        LOG.debug("Frame %s latency %.2f ms (avg %.2f, p95 %.2f)", fid, instant, avg, p95)  # Log détaillé de performance.
            except Exception:
                pass  # Tolérance complète aux erreurs pour ne jamais bloquer la pipeline.

            return True  # Succès de l’envoi.
        except Exception:
            LOG.exception("Failed to enqueue mask to outbox")  # Log complet si une erreur inattendue empêche l’envoi.
            return False  # Échec global.


    # === Helpers pour les tests et simulations ===
    def _inject_frame(self, frame: RawFrame) -> None:  # Injecte manuellement une frame dans la mailbox (utilisé pour les tests ou simulations sans PlusServer).
        try:
            self._mailbox.append(frame)  # Ajoute la frame dans la file d’entrée.
        except Exception:
            LOG.exception("Failed to inject frame into mailbox")  # Log une erreur si l’insertion échoue.

    def _drain_outbox(self) -> list:  # Vide complètement la file de sortie et renvoie son contenu (utile pour inspection en test).
        items = list(self._outbox)  # Copie tous les éléments actuellement présents.
        self._outbox.clear()  # Vide la file de sortie.
        return items  # Retourne la liste des éléments extraits.


    # === Proxys pour les statistiques et événements ===
    def update_rx_stats(self, fps: float, ts: float, bytes_count: int = 0) -> None:  # Met à jour les statistiques de réception (fréquence et taille des données).
        self.stats.update_rx(fps, ts, bytes_count)  # Appelle la méthode du collecteur GatewayStats.

    def update_tx_stats(self, fps: float, bytes_count: int = 0) -> None:  # Met à jour les statistiques d’envoi (débit de sortie).
        self.stats.update_tx(fps, bytes_count)  # Délègue la mise à jour au collecteur GatewayStats.

    def on_event(self, callback):  # Enregistre une fonction de rappel (callback) à exécuter lors d’un événement système.
        self.events.on_event(callback)  # Transmet la demande au gestionnaire d’événements EventEmitter.

    def _emit_event(self, name: str, payload: Dict[str, Any]):  # Émet un événement interne avec un nom et un contenu (dictionnaire de données).
        self.events.emit(name, payload)  # Passe la notification au système EventEmitter partagé.

