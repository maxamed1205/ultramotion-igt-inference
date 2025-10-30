"""
__main__.py
--------------------------------
Point d’entrée du simulateur Gateway mock.
Permet d’exécuter le simulateur avec :

    python -m sandbox.web_monitor.common.utils.mock_gateway_runner
"""

import sys
from pathlib import Path

# ──────────────────────────────────────────────
# 1️⃣  Préparer le contexte d'import global
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]  # Racine du projet
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ──────────────────────────────────────────────
# 2️⃣  Imports du simulateur (après que SRC soit ajouté)
# ──────────────────────────────────────────────
from sandbox.web_monitor.common.utils.mock_gateway_runner.runner import run_mock_gateway

# ──────────────────────────────────────────────
# 3️⃣  Point d’entrée exécutable
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Lancement du simulateur Gateway Mock...")
    run_mock_gateway()
