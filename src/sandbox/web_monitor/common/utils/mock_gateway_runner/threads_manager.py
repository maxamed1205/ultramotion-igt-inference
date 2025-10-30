"""
threads_manager.py
------------------
Crée et démarre les threads RX / PROC / TX du simulateur.
"""
import threading
from sandbox.web_monitor.common.utils.mock_gateway_runner.dataset_reader import read_dataset_images
from sandbox.web_monitor.common.utils.mock_gateway_runner.proc_simulator import simulate_processing
from sandbox.web_monitor.common.utils.mock_gateway_runner.tx_server import run_tx_server


def create_threads(gateway, stop_event, frame_ready, use_gpu=True, device="cpu"):
    """Crée les trois threads principaux (RX / PROC / TX) de la pipeline simulée."""

    rx_thread = threading.Thread(
        target=read_dataset_images,
        args=(gateway, stop_event, frame_ready),
        name="RX-Thread",
        daemon=True,
    )

    proc_thread = threading.Thread(
        target=simulate_processing,
        args=(gateway, stop_event, frame_ready, use_gpu, device),
        name="PROC-Thread",
        daemon=True,
    )

    tx_thread = threading.Thread(
        target=run_tx_server,
        args=(gateway, stop_event),
        name="TX-Thread",
        daemon=True,
    )

    return [rx_thread, proc_thread, tx_thread]
