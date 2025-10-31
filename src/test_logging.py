import os
import yaml
import logging
import logging.config
import signal
import threading

print("DEBUG: Starting logging test")

LOG_CFG = os.path.join(os.path.dirname(__file__), "config", "logging.yaml")
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

print(f"DEBUG: LOG_CFG path: {LOG_CFG}")
print(f"DEBUG: LOG_DIR path: {LOG_DIR}")

# create logs directory if needed (idempotent)
os.makedirs(LOG_DIR, exist_ok=True)
print("DEBUG: Logs directory created")

# Check if logging config exists
if os.path.exists(LOG_CFG):
    print("DEBUG: logging.yaml exists")
else:
    print("DEBUG: logging.yaml NOT FOUND!")
    exit(1)

# Load logging config and adjust console level according to LOG_MODE
try:
    with open(LOG_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("DEBUG: logging.yaml loaded successfully")
except Exception as e:
    print(f"DEBUG: Failed to load logging.yaml: {e}")
    exit(1)

# Allow runtime override of console verbosity: LOG_MODE=dev -> INFO, LOG_MODE=perf -> WARNING
log_mode = os.environ.get("LOG_MODE", "perf").lower()
if "handlers" in cfg and "console" in cfg["handlers"]:
    if log_mode == "dev":
        cfg["handlers"]["console"]["level"] = "INFO"
    else:
        cfg["handlers"]["console"]["level"] = "WARNING"

print("DEBUG: About to configure logging...")

try:
    logging.config.dictConfig(cfg)
    print("DEBUG: Logging configured successfully")
except Exception as e:
    print(f"DEBUG: Failed to configure logging: {e}")
    exit(1)

print("DEBUG: Testing logger...")
logger = logging.getLogger("igt.service")
logger.info("Test log message")
print("DEBUG: Logger test completed")

# Create stop event and signal handler
stop_event = threading.Event()

def shutdown_handler(signum, frame):
    print("DEBUG: Shutdown signal received")
    stop_event.set()

signal.signal(signal.SIGINT, shutdown_handler)
print("DEBUG: Signal handler set up")

print("Running... Press Ctrl+C to stop.")
stop_event.wait()
print("DEBUG: Exiting cleanly")