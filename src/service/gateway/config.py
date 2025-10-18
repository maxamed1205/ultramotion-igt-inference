"""
service.gateway.config
----------------------

Module de configuration centralisée pour le IGTGateway.
Version ultra-typée et robuste :
- lecture YAML sûre et validée,
- typage strict des champs,
- conversion bidirectionnelle dict <-> objet,
- valeurs par défaut automatiques en cas d'erreur de lecture.

Ce module est lu uniquement au démarrage (coût négligeable).
"""

from __future__ import annotations

import yaml
import os
from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict, Literal, Final, Optional
import time
import logging
LOG = logging.getLogger("igt.config")

# =============================
# 🔹 Typage strict des données
# =============================

class GatewayConfigDict(TypedDict):
    """Représente la structure exacte du fichier YAML de configuration."""
    plus_host: str
    plus_port: int
    slicer_port: int
    target_fps: float
    supervise_interval_s: float
    # optional stats block
    stats: Dict[str, Any]

# =============================
# 🔹 Classe principale
# =============================

@dataclass(slots=True)
class GatewayConfig:
    """
    Conteneur typé pour la configuration du IGTGateway.

    Champs :
      - plus_host : adresse IP / hostname du PlusServer
      - plus_port : port TCP pour réception d'images
      - slicer_port : port TCP pour envoi des masques vers Slicer
      - target_fps : cadence cible d'acquisition
      - supervise_interval_s : intervalle de supervision réseau
    """

    plus_host: str
    plus_port: int
    slicer_port: int
    target_fps: float = 25.0
    supervise_interval_s: float = 2.0
    # optional stats tuning (ex: rolling_window_size, latency_window_size)
    stats: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------
    # 🔸 Lecture YAML (robuste)
    # -----------------------------
    @classmethod
    def from_yaml(cls, path: str) -> GatewayConfig:
        """
        Charge et valide la configuration depuis un fichier YAML.

        Args:
            path : chemin du fichier YAML (ex: 'src/config/gateway.yaml')

        Returns:
            Instance GatewayConfig prête à l'emploi.

        Exceptions :
            - ValueError si des clés essentielles manquent
            - FileNotFoundError si le fichier n'existe pas
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gateway config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Validation minimale
        required_keys: Final = ("plus_host", "plus_port", "slicer_port")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required keys in config: {', '.join(missing)}")

        # Normalisation et typage forcé (garantie de cohérence)
        cfg = cls(
            plus_host=str(data["plus_host"]),
            plus_port=int(data["plus_port"]),
            slicer_port=int(data["slicer_port"]),
            target_fps=float(data.get("target_fps", 25.0)),
            supervise_interval_s=float(data.get("supervise_interval_s", 2.0)),
            stats=dict(data.get("stats", {})),
        )

        # Validation logique simple (préconditions)
        if not (1 <= cfg.plus_port <= 65535):
            raise ValueError(f"Invalid plus_port: {cfg.plus_port}")
        if not (1 <= cfg.slicer_port <= 65535):
            raise ValueError(f"Invalid slicer_port: {cfg.slicer_port}")
        if cfg.target_fps <= 0:
            raise ValueError(f"Invalid target_fps: {cfg.target_fps}")
        if cfg.supervise_interval_s <= 0:
            raise ValueError(f"Invalid supervise_interval_s: {cfg.supervise_interval_s}")

        return cfg

    # -----------------------------
    # 🔸 Conversion dict (typée)
    # -----------------------------
    def to_dict(self) -> GatewayConfigDict:
        """
        Retourne la configuration sous forme de dictionnaire typé.
        (Utile pour logging, tests unitaires, ou sérialisation JSON.)
        """
        return GatewayConfigDict(
            plus_host=self.plus_host,
            plus_port=self.plus_port,
            slicer_port=self.slicer_port,
            target_fps=self.target_fps,
            supervise_interval_s=self.supervise_interval_s,
        )

    # -----------------------------
    # 🔸 Méthodes utilitaires
    # -----------------------------
    @classmethod
    def default(cls) -> GatewayConfig:
        """Renvoie une configuration par défaut (utile pour tests)."""
        return cls("127.0.0.1", 18944, 18945, 25.0, 2.0)
    def summary(self) -> str:
        """Renvoie une chaîne lisible résumant la configuration (sans log)."""
        return (
            f"[GatewayConfig] plus_host={self.plus_host}:{self.plus_port}, "
            f"slicer_port={self.slicer_port}, fps={self.target_fps:.1f}, "
            f"interval={self.supervise_interval_s:.1f}s"
        )

    def log_summary(self) -> None:
        """Publie la configuration courante dans le système de logs interne."""
        LOG.info(self.summary())
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi

            kmsg = format_kpi({"ts": time.time(), "event": "config_loaded", "plus": f"{self.plus_host}:{self.plus_port}", "fps": f"{self.target_fps:.1f}"})
            safe_log_kpi(kmsg)
        except Exception:
            LOG.debug("KPI emission failed during config summary")
