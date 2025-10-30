import logging
import sys

def get_test_logger(name="logcollector"):
    """Logger simple et indépendant du reste du système."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    return log

# Exemple :
logger = get_test_logger()
logger.debug("LogCollector test logger initialized.")
