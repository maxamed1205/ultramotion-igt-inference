import logging
from contextlib import contextmanager

@contextmanager
def temporary_log_level(logger: logging.Logger, level: int):
    """Activate a temporary log level for a logger and restore the previous one."""
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
