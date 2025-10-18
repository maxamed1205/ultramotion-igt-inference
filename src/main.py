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


# Startup diagnostic will be logged after async selection below (so ASYNC_LOG value is available)

# Optionally enable async logging via environment variable ASYNC_LOG=1
async_enabled = os.environ.get("ASYNC_LOG", "0") in ("1", "true", "on")
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
# optional startup message
logging.getLogger("igt.service").info("Logging initialized successfully.")
# Startup diagnostic: print selected modes (now that async_enabled is known)
kpi_logging = os.environ.get("KPI_LOGGING", "1") not in ("0", "false", "False")
console_level = cfg.get("handlers", {}).get("console", {}).get("level", "WARNING")
logging.getLogger("igt.service").info(f"Startup config: LOG_MODE={log_mode} console.level={console_level} ASYNC_LOG={'on' if async_enabled else 'off'} KPI_LOGGING={'on' if kpi_logging else 'off'}")

# minimal entrypoint: import the service module or start components
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

        # start the global monitor thread and pass gateway for metrics collection
        start_monitor_thread({"interval_sec": 2.0, "gateway": gw})
        # keep running until KeyboardInterrupt
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logging.getLogger("igt.service").info("Shutdown requested")
    finally:
        try:
            if listener:
                # listener is a QueueListener object
                listener.stop()
                # attempt to join listener thread if accessible to ensure flush
                try:
                    th = getattr(listener, "thread", None) or getattr(listener, "_thread", None)
                    if th is not None:
                        th.join(timeout=2.0)
                except Exception:
                    pass
                # attempt to close underlying handlers cleanly
                try:
                    for h in getattr(listener, 'handlers', []):
                        try:
                            h.close()
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
        except Exception:
            logging.getLogger("igt.service").exception("Failed to stop log listener")
