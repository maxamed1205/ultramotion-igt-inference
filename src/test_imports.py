print("DEBUG: Starting import test")

try:
    import os
    print("DEBUG: os imported")
    import sys
    print("DEBUG: sys imported")
    import yaml
    print("DEBUG: yaml imported")
    import logging
    print("DEBUG: logging imported")
    import logging.config
    print("DEBUG: logging.config imported")
    import time
    print("DEBUG: time imported")
    
    from core.monitoring.async_logging import setup_async_logging
    print("DEBUG: setup_async_logging imported")
    
    from core.monitoring.async_logging import start_health_monitor, is_listener_alive, get_log_queue
    print("DEBUG: health monitor functions imported")
    
    from service.gateway.config import GatewayConfig
    print("DEBUG: GatewayConfig imported")
    
    from service.igthelper import IGTGateway
    print("DEBUG: IGTGateway imported")
    
    from core.monitoring.monitor import start_monitor_thread
    print("DEBUG: start_monitor_thread imported")
    
    import signal
    print("DEBUG: signal imported")
    import threading
    print("DEBUG: threading imported")
    
    print("DEBUG: All imports successful!")
    
except Exception as e:
    print(f"DEBUG: Import failed: {e}")
    import traceback
    traceback.print_exc()