import os
import time
import logging
import sys

# --- Chemin projet ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.monitoring.async_logging import setup_async_logging, get_log_queue, is_listener_alive, start_health_monitor


def test_health_monitor_emits_kpis(tmp_path):
    """Vérifie que :
    1️⃣ setup_async_logging() crée le dossier logs automatiquement s'il n'existe pas,
    2️⃣ le thread de health monitor émet bien des KPI (log_queue ou log_listener_down).
    """

    # ✅ 1. On définit un dossier temporaire pour les logs mais on NE LE CRÉE PAS
    ld = tmp_path / "logs"
    # (ne pas faire ld.mkdir() ici)

    # 2. Lancer le logging asynchrone
    q, listener = setup_async_logging(str(ld), attach_to_logger=False, yaml_cfg=None)
    assert q is not None

    # ✅ Vérifie que le dossier a bien été créé automatiquement
    assert ld.exists(), f"Le dossier {ld} aurait dû être créé automatiquement par setup_async_logging()"

    # 3. Démarrer le health monitor avec un petit intervalle
    start_health_monitor(interval=0.2, depth_warn=1, depth_crit=5, notify_after=1)

    # 4. Simule un remplissage de la file pour déclencher un KPI
    for i in range(10):
        q.put_nowait(
            logging.LogRecord(
                name="igt.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg=f"m{i}",
                args=(),
                exc_info=None,
                func=None,
            )
        )

    # 5. Laisse le temps au thread de traiter
    time.sleep(1.0)

    # 6. Le listener doit être vivant au départ
    assert is_listener_alive() is True

    # 7. On stoppe le listener pour simuler une panne
    listener.stop()
    time.sleep(1.0)

    # 8. Le fichier kpi.log doit exister et contenir au moins un événement
    kpi_log = ld / 'kpi.log'
    assert kpi_log.exists(), "Le fichier kpi.log devrait exister"
    content = kpi_log.read_text()
    assert 'event=log_queue' in content or 'event=log_listener_down' in content, "Aucun événement KPI détecté"

    # 9. Nettoyage
    try:
        listener.stop()
    except Exception:
        pass
