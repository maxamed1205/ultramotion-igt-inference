"""
env_setup.py
-------------
Prépare l'environnement d'exécution : threads NumPy, UTF-8, sys.path.
"""
import os, sys, io
from pathlib import Path

# ACTIVER MODE DEBUG pour voir les logs de latence
os.environ["LOG_MODE"] = "dev"  # dev=INFO/DEBUG, perf=WARNING


# Limiter threads BLAS
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"

# Forcer encodage UTF-8
os.system("chcp 65001 >NUL")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ajouter src/ au sys.path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
