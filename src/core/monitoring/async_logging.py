import logging  # module standard de journalisation Python
import queue  # file FIFO thread-safe, utilisée pour transporter les logs vers le listener
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler  # handlers pour logging asynchrone + rotation
from typing import Optional, Dict  # annotations de types (optionnel, dict)

# Module-level references for health checks
_log_queue: Optional[queue.Queue] = None  # référence globale vers la file de logs (pour diagnostics)
_listener_obj = None  # référence globale vers l’objet QueueListener (pour vérifier s’il est vivant)
_health_thread = None  # référence globale vers le thread de santé (surveillance du système de logs asynchrone)


def setup_async_logging(  # fonction d’installation du sous-système de logging asynchrone
    log_dir: Optional[str] = None,  # répertoire cible des fichiers de log (par défaut "logs")
    attach_to_logger: str = "igt",  # logger racine auquel attacher le QueueHandler (ex: "igt")
    yaml_cfg: Optional[Dict] = None,  # dict de la config YAML chargée (pour récupérer formatters/niveaux)
    remove_yaml_file_handlers: bool = True,  # supprime les handlers fichiers déjà présents (évite doublons)
    replace_root: bool = False,  # si True, remplace les handlers du logger racine par le QueueHandler
    create_error_handler: bool = True,  # si True, crée un handler dédié error.log (sink unique des erreurs)
):
    """Configure an asynchronous logging subsystem with a central queue.  # docstring : configure un logging asynchrone à file centrale

    Behavior:  # comportement global
    - Creates a QueueListener with file handlers for pipeline and kpi (and optionally error).  # crée un QueueListener avec handlers pipeline/kpi/(error)
    - Attaches a QueueHandler to the named logger (default 'igt').  # attache un QueueHandler au logger nommé (par défaut "igt")
    - If remove_yaml_file_handlers is True, attempts to remove RotatingFileHandler instances  # si True, retire les RotatingFileHandler déjà configurés
      from the named logger to avoid duplicate writes.  # afin d’éviter les écritures en double

    Parameters:  # paramètres
    - yaml_cfg: the dictConfig loaded YAML; used to copy formatters/levels when present.  # dict de logging.yaml pour reproduire formats/niveaux si disponibles

    Returns the started QueueListener which should be .stop()'ed on shutdown.  # retourne la file et le listener (à .stop() lors de l’arrêt)
    """
    if log_dir is None:  # si aucun répertoire n’est fourni
        log_dir = "logs"  # valeur par défaut : "logs"

    import os  # import local d’os pour créer le répertoire
    os.makedirs(log_dir, exist_ok=True)  # crée le dossier s’il n’existe pas (idempotent)


    log_queue: queue.Queue = queue.Queue(-1)  # file non bornée (-1) qui recevra tous les messages de logs

    # Determine formatters from yaml_cfg if provided, otherwise use defaults  # récupère les formatters depuis le YAML si possible
    std_formatter = None  # formatter standard (pipeline.log)
    kpi_formatter = None  # formatter KPI (kpi.log)
    if yaml_cfg and isinstance(yaml_cfg, dict):  # si une config YAML valide est fournie
        fmts = yaml_cfg.get("formatters", {})  # récupère la section "formatters" de logging.yaml
        # Build formatter objects from the format strings if present  # construit les objets logging.Formatter si des formats sont définis
        if "standard" in fmts and isinstance(fmts["standard"], dict):  # si un formatter "standard" est décrit
            std_fmt = fmts["standard"].get("format")  # lit la chaîne de format associée
            if std_fmt:  # si non vide
                std_formatter = logging.Formatter(std_fmt)  # crée l’objet Formatter pour le format standard
        if "kpi" in fmts and isinstance(fmts["kpi"], dict):  # si un formatter "kpi" est décrit
            kpi_fmt = fmts["kpi"].get("format")  # lit la chaîne de format associée
            if kpi_fmt:  # si non vide
                kpi_formatter = logging.Formatter(kpi_fmt)  # crée l’objet Formatter pour le format KPI

    # Fallback formatters  # valeurs de repli si YAML n’a pas fourni les formats
    if std_formatter is None:  # si aucun formatter standard n’a été trouvé
        std_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(processName)s/%(threadName)s | %(name)s | %(message)s")  # format par défaut complet
    if kpi_formatter is None:  # si aucun formatter KPI n’a été trouvé
        kpi_formatter = logging.Formatter("%(asctime)s | %(processName)s | %(threadName)s | %(name)s | %(message)s")  # format KPI de repli

    handler_main = RotatingFileHandler(f"{log_dir}/pipeline.log", maxBytes=10_000_000, backupCount=5)  # handler fichier avec rotation pour pipeline.log (10 Mo, 5 backups)
    handler_main.setLevel(logging.DEBUG)  # capte tous les niveaux jusqu’à DEBUG
    handler_main.setFormatter(std_formatter)  # applique le formatter standard
    # exclude ERROR+ from pipeline main (they go to error.log)  # exclut ERROR+ de pipeline.log (redirigés vers error.log)
    try:
        from core.monitoring.filters import NoErrorFilter  # filtre maison pour retirer ERROR des handlers non dédiés
        handler_main.addFilter(NoErrorFilter())  # ajoute le filtre NoErrorFilter sur pipeline.log
    except Exception:
        pass  # tolère l’absence du filtre sans casser l’initialisation

    handler_kpi = RotatingFileHandler(f"{log_dir}/kpi.log", maxBytes=5_000_000, backupCount=3)  # handler fichier pour kpi.log (5 Mo, 3 backups)
    handler_kpi.setLevel(logging.INFO)  # n’accepte que INFO et plus
    handler_kpi.setFormatter(kpi_formatter)  # applique le formatter KPI

    # Optional: JSONL KPI file (structured) controlled by KPI_JSONL env var  # option d’un sink KPI au format JSONL via variable d’environnement
    import os  # ré-import local (autorisé, inoffensif)
    kpi_jsonl_handler = None  # handler JSONL optionnel initialisé à None
    listener_handlers = [handler_main, handler_kpi]  # liste des handlers gérés par le QueueListener

    if os.getenv("KPI_JSONL", "0") not in ("0", "false", "False"):  # si KPI_JSONL activé (≠ 0/false)
        try:
            from core.monitoring.kpi import KpiJsonFormatter  # formatter spécialisé JSONL pour KPI
            kpi_jsonl_handler = RotatingFileHandler(f"{log_dir}/kpi.jsonl", maxBytes=5_000_000, backupCount=3)  # handler fichier JSONL (rotation 5 Mo, 3 backups)
            kpi_jsonl_handler.setLevel(logging.INFO)  # niveau INFO et supérieurs
            kpi_jsonl_handler.setFormatter(KpiJsonFormatter())  # applique le formatter JSONL
            listener_handlers.append(kpi_jsonl_handler)  # ajoute le handler JSONL au listener
        except Exception:
            pass  # si indisponible, on ignore sans interrompre l’installation

    # Error handler: create if requested. Prefer a single error sink under async mode.  # handler dédié aux erreurs : un seul sink en mode async
    if create_error_handler:  # si l’option de création du handler d’erreurs est activée
        handler_err = RotatingFileHandler(f"{log_dir}/error.log", maxBytes=7_340_032, backupCount=3)  # error.log (≈7 Mo, 3 backups)
        handler_err.setLevel(logging.ERROR)  # ne prend que ERROR et CRITICAL
        handler_err.setFormatter(std_formatter)  # format standard pour les erreurs
        listener_handlers.append(handler_err)  # ajoute le handler d’erreurs à la liste du listener

    listener = QueueListener(log_queue, *listener_handlers)  # crée le QueueListener avec tous les handlers de sortie
    listener.start()  # démarre le thread interne du QueueListener

    # store module-level references  # conserve des références globales pour contrôle/diagnostic
    global _log_queue, _listener_obj  # déclare l’utilisation des variables globales
    _log_queue = log_queue  # mémorise la file globale
    _listener_obj = listener  # mémorise le listener global

    queue_handler = QueueHandler(log_queue)  # crée un QueueHandler qui poussera les logs dans la file centrale

    target_logger = logging.getLogger(attach_to_logger) if attach_to_logger else logging.getLogger()  # récupère le logger cible (ou root si vide)

    # Remove file handlers defined in YAML from target logger to avoid duplication  # supprime les handlers fichiers YAML pour éviter les doublons
    if remove_yaml_file_handlers:  # seulement si demandé
        # Traverse all loggers that start with the attach_to_logger prefix and remove file handlers  # parcours des loggers descendants pour nettoyage
        prefix = attach_to_logger + "." if attach_to_logger else ""  # préfixe de la hiérarchie (ex: "igt.")
        # Remove from the direct target logger  # suppression sur le logger cible direct
        def remove_file_handlers_from_logger(lgr):  # fonction utilitaire de retrait de handlers fichiers
            for h in list(lgr.handlers):  # itère sur une copie de la liste des handlers
                try:
                    from logging.handlers import RotatingFileHandler as _RFH  # type à vérifier
                    if isinstance(h, _RFH):  # si le handler est un RotatingFileHandler
                        lgr.removeHandler(h)  # le retirer pour éviter la double écriture
                except Exception:
                    pass  # ignorer toute erreur de retrait

        remove_file_handlers_from_logger(target_logger)  # applique le nettoyage sur le logger cible

        # Also remove file handlers from all descendant loggers registered in logging.Logger.manager.loggerDict  # nettoie aussi les enfants enregistrés
        try:
            mgr = logging.Logger.manager  # accès au gestionnaire global des loggers
            for name, obj in list(mgr.loggerDict.items()):  # parcourt tous les loggers connus
                # We only want logger objects (not PlaceHolder)  # on vise les vrais loggers uniquement
                try:
                    if not name.startswith(prefix.rstrip('.')):  # ignore ceux hors du préfixe cible
                        continue  # passe au suivant
                    child = logging.getLogger(name)  # récupère le logger enfant par son nom
                    remove_file_handlers_from_logger(child)  # retire ses handlers fichiers
                except Exception:
                    pass  # ignore toute anomalie lors du parcours
        except Exception:
            pass  # si l’accès au manager échoue, on continue sans bloquer

    # Attach the QueueHandler to the target logger  # attache le QueueHandler au logger cible
    target_logger.addHandler(queue_handler)  # désormais, tous ses logs iront dans la file asynchrone

    # Optionally adjust logger level to ensure messages are passed through.  # ajuste le niveau pour garantir le passage des messages
    # If we're attaching to the root logger, ensure its level allows INFO KPIs  # si on attache au root, s’assurer qu’INFO passe
    try:
        root_logger = logging.getLogger()  # récupère le logger racine
        if target_logger is root_logger:  # si le logger cible est le root
            # If root is set to a higher level (e.g. WARNING) we still want  # si root est trop haut (WARNING+), on abaisse
            # KPI INFO messages to be processed by the queue handler.  # pour laisser passer les KPI INFO
            if root_logger.level > logging.INFO:  # test du niveau actuel
                root_logger.setLevel(logging.INFO)  # abaisse le niveau à INFO
        else:
            # For non-root target loggers preserve previous behavior: only  # pour un logger non-root, on n’écrase pas un niveau explicite
            # set level when it was explicitly unset (0).  # on fixe seulement si niveau 0 (non configuré)
            if target_logger.level == 0:  # 0 signifie “pas de niveau défini”
                target_logger.setLevel(logging.INFO)  # fixe à INFO pour garantir la remontée des KPI
    except Exception:
        # Never let logging configuration break startup  # ne jamais bloquer le démarrage sur erreur de config
        pass  # on ignore l’erreur silencieusement

    # If replace_root is requested, attach queue handler to root instead  # option : remplacer les handlers du root par le QueueHandler
    if replace_root:  # si demandé par l’appelant
        root = logging.getLogger()  # récupère le logger racine
        for h in list(root.handlers):  # itère sur copie pour modifier en sécurité
            try:
                root.removeHandler(h)  # retire chaque handler existant
            except Exception:
                pass  # ignore les erreurs de retrait
        root.addHandler(queue_handler)  # attache le QueueHandler au root (pipeline asynchrone global)

    # return both queue and the listener so callers/tests can access the queue  # retourne la file et le listener pour contrôle externe
    return log_queue, listener  # tuple (file de logs, QueueListener démarré)



def get_log_queue() -> Optional[queue.Queue]:  # retourne la file interne du système de logging asynchrone
    """Return the internal log queue (if created).  # renvoie la file interne (si elle a été créée)

    This is useful for health-checks to read queue.qsize().  # utile pour lire la taille actuelle de la file (surveillance)
    """
    return _log_queue  # renvoie la référence globale de la file (ou None si non initialisée)


def is_listener_alive() -> bool:  # vérifie si le thread QueueListener est encore actif
    """Return True if the QueueListener thread appears alive.  # retourne True si le thread du listener est vivant

    Note: uses public attributes when available.  # utilise les attributs publics du listener si disponibles
    """
    try:
        if _listener_obj is None:  # si aucun listener n’a été créé
            return False  # considéré comme inactif
        th = getattr(_listener_obj, "thread", None) or getattr(_listener_obj, "_thread", None)  # tente de récupérer le thread interne du listener (selon version Python)
        return bool(th and getattr(th, "is_alive", lambda: False)())  # renvoie True si le thread existe et est vivant
    except Exception:
        return False  # en cas d’erreur, considère que le listener n’est pas vivant


def start_health_monitor(interval: float = 5.0, depth_warn: int = 1000, depth_crit: int = 5000, notify_after: int = 1):  # démarre un thread de surveillance continue
    """Start a background thread that monitors the log queue and listener liveness.  # démarre un thread en arrière-plan qui surveille la file et le listener

    This monitor runs periodically and emits structured KPIs and log warnings  # il s’exécute périodiquement et émet des KPI structurés + alertes
    to help detect and alert on issues with the asynchronous logging subsystem.  # pour détecter les anomalies du système de logging asynchrone

    Args:  # paramètres d’entrée
        interval: seconds between checks (float). Lower values increase sensitivity.  # intervalle entre deux vérifications
        depth_warn: queue depth threshold to emit a WARNING.  # seuil d’alerte (taille de file)
        depth_crit: queue depth threshold to emit an ERROR.  # seuil critique (taille de file)
        notify_after: number of consecutive dead checks before emitting log_listener_down KPI.  # nb de détections consécutives avant alerte "listener down"

    Behavior:  # comportement détaillé
        - Every `interval` seconds the monitor reads the internal log queue size  # toutes les N secondes : lit la taille de la file interne
          and emits a KPI line: kpi ts=<float> event=log_queue depth=<n> dropped=<n>  # envoie une ligne KPI sur la profondeur de la file
          where `dropped` is a best-effort read of the global drop counter (if available).  # le champ dropped provient d’un compteur global (si présent)
        - If the QueueListener appears not alive for `notify_after` consecutive checks  # si le listener est mort plusieurs fois de suite
          the monitor emits kpi ts=... event=log_listener_down consecutive=<n>  # émet un KPI "listener down"
          and logs a WARNING on logger `igt.monitor`.  # et journalise un avertissement
        - If the queue depth exceeds `depth_warn` a WARNING is logged; if it exceeds  # si la file est trop pleine
          `depth_crit` an ERROR is logged.  # log ERROR au-delà du seuil critique

    Restart semantics:  # politique de redémarrage
        - The health monitor is a daemon thread; it does not attempt to auto-restart the  # c’est un thread démon, il ne redémarre pas le listener
          QueueListener itself but will continue to emit KPIs describing queue depth and  # il se contente de signaler l’état via KPI
          listener liveness. External orchestration (supervisor) should react to these KPIs.  # la supervision externe doit réagir

    KPIs emitted:  # types de KPI générés
        - event=log_queue: depth=<int> dropped=<int>  # KPI de profondeur de file
        - event=log_listener_down: consecutive=<int>  # KPI d’état du listener

    Note: KPI emission is best-effort and must never raise exceptions into the monitor loop.  # le thread ne doit jamais planter même si une exception survient
    
    KPIs produced by this monitor are written via the KPI logger (typically `logs/kpi.log`).  # les KPI sont envoyés dans logs/kpi.log
    When the optional KPI JSONL sink is enabled (KPI_JSONL=1) a structured `logs/kpi.jsonl` file  # si KPI_JSONL=1, un fichier structuré est aussi écrit
    is also written by the async logging subsystem when ASYNC_LOG=1.  # (seulement si le mode async est activé)
    """

    global _health_thread  # déclare la variable globale du thread de santé

    def _monitor_loop():  # boucle interne exécutée dans le thread de surveillance
        import time  # pour la temporisation
        from logging import getLogger  # récupération du logger interne

        log = getLogger("igt.monitor")  # logger dédié aux messages du moniteur
        consecutive_dead = 0  # compteur de détections consécutives de listener mort
        while True:  # boucle infinie (thread démon)
            try:
                q = get_log_queue()  # récupère la file de logs
                depth = q.qsize() if q is not None else 0  # lit sa profondeur (ou 0 si non créée)
                # emit KPI for queue depth and drops  # émet un KPI sur la profondeur + pertes
                try:
                    from core.monitoring.kpi import safe_log_kpi, format_kpi, get_drop_count  # fonctions utilitaires KPI

                    # include current drop counters if available  # ajoute le compteur global des drops si dispo
                    drops = 0  # valeur par défaut
                    try:
                        drops = int(get_drop_count("acq.drop_total"))  # tente de lire le compteur global "acq.drop_total"
                    except Exception:
                        drops = 0  # en cas d’échec, retombe à 0

                    kmsg = format_kpi({"ts": time.time(), "event": "log_queue", "depth": depth, "dropped": drops})  # formatte le KPI log_queue
                    safe_log_kpi(kmsg)  # envoie le KPI au logger "igt.kpi"
                except Exception:
                    log.debug("Failed to emit log_queue KPI")  # message debug si l’émission échoue

                # check listener liveness  # vérifie la vitalité du listener
                alive = is_listener_alive()  # appelle la fonction précédente
                if not alive:  # si le listener est mort
                    consecutive_dead += 1  # incrémente le compteur
                    if consecutive_dead >= notify_after:  # si dépasse le seuil de notifications
                        # warn and emit KPI once per check interval  # alerte + KPI "listener down"
                        log.warning("Async log listener appears down (consecutive=%d)", consecutive_dead)  # warning dans logs
                        try:
                            from core.monitoring.kpi import safe_log_kpi, format_kpi  # réimport minimal
                            kmsg = format_kpi({"ts": time.time(), "event": "log_listener_down", "consecutive": consecutive_dead})  # formate KPI d’état
                            safe_log_kpi(kmsg)  # envoie le KPI
                        except Exception:
                            log.debug("Failed to emit log_listener_down KPI")  # log debug si l’émission échoue
                else:
                    consecutive_dead = 0  # si listener vivant, reset du compteur

                # thresholds -> warning/error via monitor logger  # compare la taille de file aux seuils
                if depth >= depth_crit:  # si dépasse seuil critique
                    log.error("Log queue depth critical=%d", depth)  # log ERROR
                elif depth >= depth_warn:  # sinon, si dépasse seuil d’avertissement
                    log.warning("Log queue depth warning=%d", depth)  # log WARNING

                time.sleep(interval)  # attend avant la prochaine vérification
            except Exception:
                try:
                    log.exception("Exception in async health monitor loop")  # logge l’exception (sans casser la boucle)
                except Exception:
                    pass  # ignore toute erreur de logging
                time.sleep(interval)  # pause avant reprise pour éviter boucle folle

    if _health_thread is None:  # ne crée le thread qu’une seule fois
        import threading  # module de threads
        _health_thread = threading.Thread(target=_monitor_loop, name="AsyncLogHealth", daemon=True)  # crée un thread démon pour la boucle de surveillance
        _health_thread.start()  # lance le thread
    return _health_thread  # retourne la référence du thread (pour tests ou suivi)

