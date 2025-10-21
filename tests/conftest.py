import sys
from pathlib import Path

# Ensure repository's src/ is on sys.path for tests
ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
