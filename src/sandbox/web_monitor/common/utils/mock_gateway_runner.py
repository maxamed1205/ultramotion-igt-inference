
import sys
import time
import threading
import signal
import numpy as np
from pathlib import Path
from PIL import Image
import glob


# ──────────────────────────────────────────────
#  Imports pipeline réelle
# ──────────────────────────────────────────────
import torch  # Import torch pour les opérations GPU
from service.gateway.manager import IGTGateway
from service.slicer_server import run_slicer_server
from core.types import RawFrame, FrameMeta, Pose
from core.preprocessing.cpu_to_gpu import (
    init_transfer_runtime,
    prepare_frame_for_gpu,
)

# ──────────────────────────────────────────────
#  Logger asynchrone (sera configuré dans __main__)
# ──────────────────────────────────────────────
import logging.config, yaml
from core.monitoring import async_logging

LOG = logging.getLogger("igt.gateway.test")



