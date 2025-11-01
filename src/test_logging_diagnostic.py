import os
import sys
import yaml
import logging
import logging.config
import time

# Configuration de base
LOG_CFG = os.path.join(os.path.dirname(__file__), "config", "logging.yaml")
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

print(f"LOG_CFG: {LOG_CFG}")
print(f"LOG_DIR: {LOG_DIR}")

# Créer le répertoire de logs
os.makedirs(LOG_DIR, exist_ok=True)

# Test 1: Logging avec config YAML seulement (sans async)
print("\n=== TEST 1: YAML CONFIG SEULEMENT ===")

with open(LOG_CFG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Ajuster le mode console
log_mode = "dev"  # Force mode dev pour voir les logs
if "handlers" in cfg and "console" in cfg["handlers"]:
    cfg["handlers"]["console"]["level"] = "INFO"

# Appliquer la config
logging.config.dictConfig(cfg)

# Test des logs
logger = logging.getLogger("igt.service")
print(f"Logger igt.service handlers: {logger.handlers}")
print(f"Logger igt.service level: {logger.level}")

logger.info("TEST 1: Message INFO depuis igt.service (YAML seulement)")
logger.warning("TEST 1: Message WARNING depuis igt.service (YAML seulement)")
logger.error("TEST 1: Message ERROR depuis igt.service (YAML seulement)")

# Attendre un peu et vérifier les fichiers
time.sleep(1)
print("\nContenu des fichiers après TEST 1:")

files = ["pipeline.log", "error.log", "kpi.log"]
for filename in files:
    filepath = os.path.join(LOG_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"{filename}: {'VIDE' if not content.strip() else f'{len(content.split())} lignes'}")
            if content.strip():
                print(f"  Contenu: {content.strip()[:100]}...")
    else:
        print(f"{filename}: N'EXISTE PAS")