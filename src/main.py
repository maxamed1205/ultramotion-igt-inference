import os
import sys
import yaml
import logging
import logging.config
import time
from core.monitoring.async_logging import setup_async_logging
from core.monitoring.async_logging import start_health_monitor, is_listener_alive, get_log_queue
from service.gateway.config import GatewayConfig
from service.igthelper import IGTGateway
from core.monitoring.monitor import start_monitor_thread
import signal
import threading

LOG_CFG = os.path.join(os.path.dirname(__file__), "config", "logging.yaml")
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

os.makedirs(LOG_DIR, exist_ok=True) # Créer un répertoire de logs si nécessaire (idempotent)

# Fonction pour supprimer les fichiers dans le dossier logs
def clear_log_directory(log_dir):
    """Supprimer tous les fichiers dans le répertoire des logs"""
    if os.path.exists(log_dir):
        for file_name in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"Fichier supprimé: {file_path}")
                elif os.path.isdir(file_path):
                    # Supprimer les fichiers dans les sous-dossiers
                    for subfile in os.listdir(file_path):
                        os.remove(os.path.join(file_path, subfile))
                        logging.info(f"Fichier supprimé dans sous-dossier: {os.path.join(file_path, subfile)}")
            except Exception as e:
                logging.error(f"Erreur lors de la suppression de {file_path}: {e}")
    else:
        logging.warning(f"Le répertoire {log_dir} n'existe pas")

# Supprimer les fichiers de logs avant chaque test
clear_log_directory(LOG_DIR)

with open(LOG_CFG, "r", encoding="utf-8") as f:# Charger la configuration de journalisation et ajuster le niveau de la console en fonction de LOG_MODE
    cfg = yaml.safe_load(f)

# Fix file paths to be absolute
if "handlers" in cfg:
    for handler_name, handler_config in cfg["handlers"].items():
        if "filename" in handler_config and handler_config["filename"].startswith("logs/"):
            relative_path = handler_config["filename"]
            absolute_path = os.path.join(LOG_DIR, relative_path.replace("logs/", ""))
            cfg["handlers"][handler_name]["filename"] = absolute_path

log_mode = os.environ.get("LOG_MODE", "perf").lower() # Autoriser la modification dynamique du niveau de verbosité de la console : LOG_MODE=dev -> INFO, LOG_MODE=perf -> WARNING
if "handlers" in cfg and "console" in cfg["handlers"]:
    if log_mode == "dev":
        cfg["handlers"]["console"]["level"] = "INFO"
    else:
        cfg["handlers"]["console"]["level"] = "WARNING"

logging.config.dictConfig(cfg)

# Optionally enable async logging via environment variable ASYNC_LOG=1
os.environ["ASYNC_DEBUG"] = "1"
os.environ["ASYNC_DEBUG_STEP"] = "after_attach"
# async_enabled = os.environ.get("ASYNC_LOG", "0") in ("1", "true", "on")
async_enabled = True  # Par défaut, activons l'async logging
listener = None
if async_enabled:
    try:
        log_queue, listener = setup_async_logging( # Configurer un écouteur asynchrone qui reproduit les formateurs YAML et écrit les pipelines/KPI
            log_dir=os.path.abspath(LOG_DIR),
            attach_to_logger="igt",
            yaml_cfg=cfg,
            remove_yaml_file_handlers=True,
            # replace_root=False,
            create_error_handler=True,  # si True, crée un handler dédié error.log (sink unique des erreurs)

        )
        # start health monitor for the async subsystem
        try:
            start_health_monitor(interval=5.0)
        except Exception:
            logging.getLogger("igt.service").debug("Failed to start async health monitor")
    except Exception:
        logging.getLogger("igt.service").exception("Failed to enable async logging; falling back to YAML file handlers")
        listener = None
else:
    logging.getLogger("igt.service").info(f"Async logging disabled; LOG_MODE={log_mode} ASYNC_LOG=off")

# exit()

# Create a global stop event
stop_event = threading.Event()

# Set up signal handler for graceful shutdown (e.g., Ctrl+C)
def shutdown_handler(signum, frame):
    """Handle shutdown signal gracefully"""
    logging.getLogger("igt.service").info("Shutdown requested")
    stop_event.set()  # Set the event to stop the main loop

# Main entry point
if __name__ == "__main__":
    logging.getLogger("igt.service").info("Ultramotion IGT Inference service (logging configured)")
    try:
        # Load gateway configuration and start the gateway
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "gateway.yaml")
        try:
            gw_cfg = GatewayConfig.from_yaml(cfg_path)
        except Exception:
            logging.getLogger("igt.service").exception("Failed to load gateway config; using defaults")
            gw_cfg = GatewayConfig(plus_host="127.0.0.1", plus_port=18944, slicer_port=18945)
        
        gw = IGTGateway(gw_cfg)
        gw.start()
        start_monitor_thread({"interval_sec": 2.0, "gateway": gw})  # start the global monitor thread and pass gateway for metrics collection
        
        # Set up signal handler for graceful shutdown (e.g., Ctrl+C)
        signal.signal(signal.SIGINT, shutdown_handler)

        # Main loop to wait for shutdown signal (Windows-compatible)
        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=0.1)  # Short timeout to allow KeyboardInterrupt
        except KeyboardInterrupt:
            stop_event.set()
        
        logging.getLogger("igt.service").info("Service stopped.")

    except Exception as e:
        logging.getLogger("igt.service").exception(f"Unexpected error: {e}")
    finally:
        if listener:
            listener.stop()  # Ensure the listener stops cleanly
