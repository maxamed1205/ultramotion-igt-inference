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

# create logs directory if needed (idempotent)
os.makedirs(LOG_DIR, exist_ok=True)

# Load logging config and adjust console level according to LOG_MODE
with open(LOG_CFG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Allow runtime override of console verbosity: LOG_MODE=dev -> INFO, LOG_MODE=perf -> WARNING
log_mode = os.environ.get("LOG_MODE", "perf").lower()
if "handlers" in cfg and "console" in cfg["handlers"]:
    if log_mode == "dev":
        cfg["handlers"]["console"]["level"] = "INFO"
    else:
        cfg["handlers"]["console"]["level"] = "WARNING"

logging.config.dictConfig(cfg)

# Optionally enable async logging via environment variable ASYNC_LOG=1
# os.environ["ASYNC_LOG"] = "1"
# async_enabled = os.environ.get("ASYNC_LOG", "0") in ("1", "true", "on")
async_enabled = True  # Par défaut, activons l'async logging
listener = None
if async_enabled:
    try:
        # Setup an async listener that mirrors YAML formatters and writes pipeline/kpi
        log_queue, listener = setup_async_logging(
            log_dir=os.path.abspath(LOG_DIR),
            attach_to_logger="igt",
            yaml_cfg=cfg,
            remove_yaml_file_handlers=True,
            replace_root=False,
        )
        # start health monitor for the async subsystem
        try:
            start_health_monitor(interval=5.0)
        except Exception:
            logging.getLogger("igt.service").debug("Failed to start async health monitor")
        logging.getLogger("igt.service").info(f"Async logging enabled; LOG_MODE={log_mode} ASYNC_LOG=on")
    except Exception:
        logging.getLogger("igt.service").exception("Failed to enable async logging; falling back to YAML file handlers")
        listener = None
else:
    logging.getLogger("igt.service").info(f"Async logging disabled; LOG_MODE={log_mode} ASYNC_LOG=off")

logging.getLogger("igt.service").info("Logging initialized successfully.")

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
