"""
run_web_monitor.py
------------------
Point d’entrée du module Web Monitor.

Rôle :
- Lire la configuration YAML (dashboard.yaml)
- Initialiser le logger local 'igt.dashboard'
- Garantir la compatibilité UTF-8 via async_logging (import sans exécution de listeners)
- Lancer le serveur FastAPI en mode 'dashboard'
- Préparer le mode 'pipeline' (placeholder)

Ce fichier est volontairement autonome et documenté en français.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn

# IMPORTANT : import du patch d'async_logging pour garantir la compatibilité
# d'encodage sous Windows. Ne pas appeler de fonctions d'initialisation ici.

# Ajoute le dossier 'src' au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

from core.monitoring import async_logging # type: ignore


# Tentative d'import de la fonction utilitaire du module common; si elle
# n'existe pas encore, on définit un stub qui lèvera une erreur lisible au
# moment du chargement de la configuration (load_configuration gèrera l'erreur).
try:
    from sandbox.web_monitor.common.config_loader import load_dashboard_config
except Exception:
    def load_dashboard_config(path: str) -> Any:  # type: ignore
        raise RuntimeError(
            "La fonction 'load_dashboard_config' n'est pas disponible dans sandbox.web_monitor.common.config_loader."
        )

try:
    from sandbox.web_monitor.common.app.app_factory import create_app
except Exception:
    def create_app(cfg: Any):  # type: ignore
        raise RuntimeError(
            "La fonction 'create_app' n'est pas disponible dans sandbox.web_monitor.common.app.app_factory."
        )



def setup_dashboard_logger() -> logging.Logger:
    """
    Initialise le logger local 'igt.dashboard'.

    - Écrit uniquement sur la console (stdout)
    - N'essaie pas de modifier ou de réinitialiser le sous-système global async_logging
    - Retourne le logger configuré
    """
    logger = logging.getLogger("igt.dashboard")
    logger.setLevel(logging.INFO)

    # Éviter d'ajouter plusieurs handlers si le logger est déjà configuré
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultramotion Web Monitor")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "pipeline"],
        default="dashboard",
        help="Mode d'exécution : dashboard ou pipeline",
    )
    parser.add_argument(
        "--config",
        default="src/sandbox/web_monitor/config/dashboard.yaml",
        help="Chemin vers le fichier de configuration (relatif au dossier web_monitor)",
    )
    return parser.parse_args()


def load_configuration(path: str, log: logging.Logger) -> Any:
    """
    Charge la configuration YAML via la fonction fournie par
    `common.config_loader.load_dashboard_config`.

    En cas d'erreur, logge et stoppe le processus proprement.
    """
    try:
        cfg = load_dashboard_config(path)
        # log.info(f"Configuration chargée depuis {path}")
        return cfg
    except Exception as e:  # pragma: no cover - gestion d'erreur runtime
        log.error(f"Erreur lors du chargement de la configuration : {e}")
        sys.exit(1)


async def launch_dashboard(cfg: Any, log: logging.Logger) -> None:
    """
    Lance le serveur FastAPI du Web Monitor via uvicorn.

    Cette fonction est asynchrone pour faciliter l'intégration future
    (graceful shutdown, tâches background, etc.).
    """
    # create_app doit retourner une instance ASGI (FastAPI)
    app = create_app(cfg)

    # On lit les paramètres d'hôte/port depuis la config attendue
    host = cfg.get("dashboard", {}).get("host", "0.0.0.0")
    port = int(cfg.get("dashboard", {}).get("port", 8050))

    log.info(f"Démarrage du Web Monitor sur {host}:{port}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    # Démarre le serveur (bloquant tant que le serveur tourne)
    await server.serve()


def main() -> None:
    args = parse_arguments()
    log = setup_dashboard_logger()

    log.info(f"Lancement du Web Monitor en mode: {args.mode}")

    # Charge la configuration
    cfg = load_configuration(args.config, log)

    if args.mode == "dashboard":
        # Lance l'application FastAPI
        try:
            asyncio.run(launch_dashboard(cfg, log))
        except KeyboardInterrupt:
            # Propagé pour être géré par le bloc supérieur
            raise
        except Exception as e:
            log.exception(f"Erreur lors du lancement du dashboard: {e}")
            sys.exit(1)
    elif args.mode == "pipeline":
        log.warning("Mode pipeline non implémenté (placeholder).")
    else:
        log.error(f"Mode inconnu: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Message simple et lisible pour l'utilisateur
        print("\nArrêt du Web Monitor (CTRL+C).")
    except Exception as e:
        logging.getLogger("igt.dashboard").exception(f"Erreur critique: {e}")
        sys.exit(1)


