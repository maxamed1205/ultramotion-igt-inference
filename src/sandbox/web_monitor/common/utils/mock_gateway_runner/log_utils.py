"""
log_utils.py
------------
Utilitaires pour la gestion des logs dans le simulateur Gateway :
- Nettoyage des anciens fichiers de log
- Configuration du logging asynchrone via logging.yaml
"""

import logging
import yaml
from pathlib import Path
from core.monitoring import async_logging  # système interne de logs asynchrones

# ──────────────────────────────────────────────
#  Constantes globales
# ──────────────────────────────────────────────
# On est dans src/sandbox/web_monitor/common/utils/mock_gateway_runner/
# → remonter 3 niveaux pour atteindre src/
ROOT = Path(__file__).resolve().parents[5]
LOG = logging.getLogger("igt.mock.log_utils")


# ──────────────────────────────────────────────
#  🧹 Nettoyage des logs avant test
# ──────────────────────────────────────────────
def clean_old_logs():
    """Supprime tous les fichiers .log dans logs/ avant de démarrer un nouveau test."""
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        LOG.warning(f"Dossier de logs introuvable : {logs_dir}")
        return

    deleted_count = 0
    for log_file in logs_dir.glob("*.log"):
        try:
            log_file.unlink()
            deleted_count += 1
        except PermissionError:
            LOG.debug(f"Fichier verrouillé (ignoré) : {log_file}")
        except Exception as e:
            LOG.debug(f"Erreur suppression {log_file}: {e}")

    if deleted_count > 0:
        print(f"[CLEAN] {deleted_count} fichier(s) log supprimé(s)")
    else:
        # print("[CLEAN] Aucun log supprimé (ou fichiers verrouillés)")
        pass

# ──────────────────────────────────────────────
#  ⚙️ Configuration du logging asynchrone
# ──────────────────────────────────────────────
def setup_logging():
    """Configure le logging asynchrone avec src/config/logging.yaml."""
    LOG_CFG = ROOT / "config" / "logging.yaml"  # ✅ correction ici

    if not LOG_CFG.exists():
        LOG.warning(f"[LOG] Fichier introuvable : {LOG_CFG}")
        logging.basicConfig(level=logging.INFO)
        print("⚠️ logging.yaml non trouvé — fallback console activé.")
        return

    try:
        # Charger la configuration YAML et initialiser le logging asynchrone
        with open(LOG_CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            async_logging.setup_async_logging(yaml_cfg=cfg)
            async_logging.start_health_monitor()
            LOG.info("[LOG] Configuration asynchrone initialisée ✅")

        # 🔇 Suppression TOTALE de tous les StreamHandlers
        removed = 0
        for name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger):
                for h in list(logger.handlers):
                    if isinstance(h, logging.StreamHandler):
                        logger.removeHandler(h)
                        removed += 1
        for h in list(logging.root.handlers):
            if isinstance(h, logging.StreamHandler):
                logging.root.removeHandler(h)
                removed += 1

        LOG.info(f"[MOCK] Mode silencieux console activé — {removed} StreamHandler(s) supprimé(s)")

        # 🧱 Protection : empêcher tout ajout futur de StreamHandler
        _original_addHandler = logging.Logger.addHandler

        def _patched_addHandler(self, hdlr, *args, **kwargs):
            if isinstance(hdlr, logging.StreamHandler):
                return  # ignorer silencieusement
            return _original_addHandler(self, hdlr, *args, **kwargs)

        logging.Logger.addHandler = _patched_addHandler
        LOG.info("[MOCK] Protection active : aucun StreamHandler ne sera recréé")

        # 🔍 DIAGNOSTIC silencieux (écrit dans logs/_diagnostic_handlers.txt)
        try:
            diag_path = ROOT / "logs" / "_diagnostic_handlers.txt"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write("===== DIAGNOSTIC LOGGING (POST-CONFIG) =====\n")
                for name, logger in logging.root.manager.loggerDict.items():
                    if isinstance(logger, logging.Logger):
                        for h in logger.handlers:
                            f.write(f"[{name}] handler={type(h).__name__} level={h.level}\n")
                for h in logging.root.handlers:
                    f.write(f"[ROOT] handler={type(h).__name__} level={h.level}\n")
                f.write("=============================================\n")
            LOG.info(f"[DIAG] Rapport de loggers sauvegardé dans {diag_path}")
        except Exception as diag_err:
            LOG.error(f"[DIAG] Erreur diagnostic logging: {diag_err}")

    except Exception as e:
        LOG.error(f"[LOG] Erreur de configuration du logging : {e}")
        logging.basicConfig(level=logging.INFO)
