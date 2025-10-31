# === Standard Library ===
import logging

# === Project Modules ===
from service.gateway.manager import IGTGateway
from core.types import RawFrame, FrameMeta, Pose

from service.gateway.config import GatewayConfig

from core.monitoring.async_logging import setup_async_logging
from core.monitoring.filters import PerfFilter, NoErrorFilter
from core.monitoring.kpi import increment_drops, safe_log_kpi, format_kpi  # Si nécessaire

# Charger la configuration depuis un fichier YAML
# config = GatewayConfig.from_yaml("src\config\gateway.yaml")
# Instancier IGTGateway avec l'objet GatewayConfig
# gateway = IGTGateway(config)

# Initialisation avec des arguments explicites
gateway = IGTGateway(
    "localhost",  # Adresse de PlusServer (localhost pour les tests)
    8050,         # Port d'entrée pour PlusServer
    8050,         # Port de sortie pour 3D Slicer
    target_fps=25.0,  # Cadence cible
    supervise_interval_s=2.0  # Intervalle de supervision
)

# Démarrer la passerelle
gateway.start()

# Optionnellement, vérifier l'état
status = gateway.get_status()
logging.info(f"Gateway status: {status}")