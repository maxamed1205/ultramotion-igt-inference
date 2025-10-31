"""
tailer.py
----------
Thread qui suit un fichier en continu, façon `tail -f`,
et met chaque nouvelle ligne dans une Queue.
"""

import os
import time
import threading
from . import logger


class FileTailer(threading.Thread):
    """Lit un fichier en continu et envoie les lignes dans une queue."""

    def __init__(self, filepath, queue, poll_interval=0.05):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.queue = queue
        self.poll_interval = poll_interval
        self.running = threading.Event()  # ✅ thread-safe flag
        self.line_count = 0               # compteur de lignes lues

    def run(self):
        self.running.set()
        logger.info(f"[Tailer] 🚀 Démarrage du tailer pour {os.path.basename(self.filepath)}")

        try:
            with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
                # 🟢 LIRE DÈS LE DÉBUT (mode replay + live continu)
                logger.debug(f"[Tailer] Lecture initiale complète de {self.filepath}")
                for line in f:
                    self.queue.put(line)

                # ➕ Ensuite, on passe en mode “tail -f”
                f.seek(0, os.SEEK_END)
                logger.debug(f"[Tailer] Position initiale fin de fichier ({f.tell()} octets)")

                # 🔁 Lecture continue
                while self.running.is_set():
                    line = f.readline()
                    if line:
                        self.queue.put(line)
                    else:
                        time.sleep(self.poll_interval)

        except FileNotFoundError:
            logger.warning(f"[Tailer] ⚠️ Fichier introuvable: {self.filepath}")

        except Exception as e:
            logger.error(f"[Tailer] 💥 Erreur sur {self.filepath}: {e}")

        finally:
            logger.info(f"[Tailer] 🛑 Arrêt propre du tailer {os.path.basename(self.filepath)}")

    def stop(self):
        """Demande l'arrêt du tailer."""
        self.running.clear()
