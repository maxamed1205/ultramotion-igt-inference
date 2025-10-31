import sys
import os

print("DEBUG: Script starting")
print(f"DEBUG: Python version: {sys.version}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: Script location: {__file__}")

try:
    print("DEBUG: Testing basic imports...")
    import yaml
    print("DEBUG: yaml imported")
    
    import logging
    print("DEBUG: logging imported")
    
    import threading
    print("DEBUG: threading imported")
    
    import signal
    print("DEBUG: signal imported")
    
    print("DEBUG: Testing file paths...")
    LOG_CFG = os.path.join(os.path.dirname(__file__), "config", "logging.yaml")
    print(f"DEBUG: LOG_CFG path: {LOG_CFG}")
    print(f"DEBUG: LOG_CFG exists: {os.path.exists(LOG_CFG)}")
    
    if os.path.exists(LOG_CFG):
        print("DEBUG: Testing YAML load...")
        with open(LOG_CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("DEBUG: YAML loaded successfully")
        
        print("DEBUG: Testing logging config...")
        logging.config.dictConfig(cfg)
        print("DEBUG: Logging configured")
        
        logger = logging.getLogger("igt.service")
        logger.info("Test message")
        print("DEBUG: Logger test successful")
    
    print("DEBUG: All basic tests passed")
    
except Exception as e:
    print(f"DEBUG: Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("DEBUG: Script ending")