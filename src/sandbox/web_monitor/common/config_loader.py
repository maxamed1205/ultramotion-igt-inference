"""
Module : common.config_loader
-----------------------------
Charge les fichiers YAML du Web Monitor, résout les chemins absolus
et prépare un dictionnaire de configuration cohérent.

Cette fonction est appelée dès le lancement du dashboard
depuis run_web_monitor.py (Phase 1B).
"""

from pathlib import Path
import yaml
import logging

def load_dashboard_config(path: str) -> dict:
    """
    Charge et valide le fichier YAML de configuration du Web Monitor.

    Args:
        path (str): Chemin du fichier YAML (relatif ou absolu).
    Returns:
        dict: Dictionnaire Python prêt à l'emploi.
    Raises:
        FileNotFoundError, ValueError, yaml.YAMLError
    """
    log = logging.getLogger("igt.dashboard")

    # Résolution du chemin absolu
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {cfg_path}")

    # Lecture du YAML
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "dashboard" not in cfg:
        raise ValueError("Le fichier YAML est invalide ou ne contient pas la clé 'dashboard'.")

    # Normalisation des chemins de logs (absolus)
    base_dir = cfg_path.parent.parent  # on remonte depuis /config/
    log_paths = cfg["dashboard"].get("log_paths", {})

    for key, rel in log_paths.items():
        abs_path = (base_dir / rel).resolve()
        cfg["dashboard"]["log_paths"][key] = str(abs_path)

    # Valeurs par défaut
    cfg["dashboard"].setdefault("host", "0.0.0.0")
    cfg["dashboard"].setdefault("port", 8050)
    cfg["dashboard"].setdefault("update_interval", 1.0)
    cfg["dashboard"].setdefault("mode", "live")

    log.info(f"Configuration chargée avec succès depuis {cfg_path}")
    return cfg
