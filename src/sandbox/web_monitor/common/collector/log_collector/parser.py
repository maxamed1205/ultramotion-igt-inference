"""
parser.py
----------
Analyse les lignes des logs (regex précompilées et dispatch rapide).
Renvoie un dictionnaire minimal pour agrégation.
"""

import re
from datetime import datetime

from pyparsing import line
from . import logger


class LogParser:
    """Détecte et extrait les événements des logs pipeline.log et kpi.log."""

    def __init__(self):
        # ─── Regex précompilées (accès ultra-rapide)
        # Use case-insensitive matching for the simple frame-id patterns so
        # both "Frame #001" and "frame #001" are recognised in logs.
        self.patterns = {
            # Événements principaux
            "rx": re.compile(r"\[DATASET-RX\].*frame\s*#?(\d+)", re.I),
            "proc": re.compile(r"\[PROC-SIM\].*frame\s*#?(\d+)", re.I),
            "tx": re.compile(r"\[TX(?:-SIM)?\].*Frame\s*#?(\d+)", re.I),

            # Transferts GPU (copy_async)
            "copy_async": re.compile(
                r"event=copy_async.*norm_ms=([\d.]+).*pin_ms=([\d.]+).*"
                r"copy_ms=([\d.]+).*total_ms=([\d.]+).*frame=(\d+)"
            ),

            # Bloc multi-lignes "Inter-stage latencies"
            "interstage": re.compile(
                r"RX → CPU-to-GPU:\s*([\d.]+)ms.*?"
                r"CPU-to-GPU → PROC:\s*([\d.]+)ms.*?"
                r"PROC → GPU-to-CPU:\s*([\d.]+)ms.*?"
                r"Total processing:\s*([\d.]+)",
                re.S,
            ),

            # En-tête du bloc Inter-stage (contient l’ID de frame)
            "interstage_header": re.compile(r"Inter-stage latencies\s*#?(\d+)", re.I),
        }


        self.last_frame_id = None       # garde en mémoire le dernier frame_id
        self._pending_interstage = ""   # buffer multi-lignes

    # ------------------------------------------------------------------ #
    def parse_line(self, line: str, source: str):
        """Dispatch principal rapide (pipeline/kpi)."""
        parsed = None 
        line = line.strip()
        if not line:
            return None

        if "Inter-stage" in line:
            self._pending_interstage = line + "\n"

            # 🔎 Tente de récupérer le numéro de frame dans la même ligne
            header_match = self.patterns["interstage_header"].search(line)
            if header_match:
                self.last_frame_id = int(header_match.group(1))
                logger.debug(f"[Parser] 🧩 Début bloc interstage détecté pour frame #{self.last_frame_id}")
            else:
                logger.debug("[Parser] 🧩 Début bloc interstage détecté (sans ID explicite)")
            return None

        return parsed



    # ------------------------------------------------------------------ #
    def _parse_event(self, line: str, tag: str):
        """Parse RX/PROC/TX lines with timestamp."""
        m = self.patterns[tag].search(line)
        if not m:
            return None
        fid = int(m.group(1))
        self.last_frame_id = fid
        ts = self._extract_ts(line)
        # logger.debug(f"[Parser] {tag.upper()} détecté frame #{fid}")
        return {"frame_id": fid, "event": tag, "ts": ts}

    # ------------------------------------------------------------------ #
    def _parse_copy_async(self, line: str):
        m = self.patterns["copy_async"].search(line)
        if not m:
            return None
        norm_ms, pin_ms, copy_ms, total_ms, fid = m.groups()
        fid = int(fid)
        self.last_frame_id = fid
        # logger.debug(f"[Parser] copy_async détecté frame #{fid}")
        return {
            "frame_id": fid,
            "event": "copy_async",
            "latencies": {
                "norm_ms": float(norm_ms),
                "pin_ms": float(pin_ms),
                "copy_ms": float(copy_ms),
                "cpu_gpu": float(total_ms),
            },
        }

    # ------------------------------------------------------------------ #
    def _parse_interstage(self, block: str):
        """Parse un bloc complet 'Inter-stage latencies' sur plusieurs lignes."""
        m = self.patterns["interstage"].search(block)
        # if not m:
        #     logger.debug("[Parser] Bloc interstage non reconnu")
        #     return None
        rx_cpu, cpu_gpu, proc_gpu, total = map(float, m.groups())

        logger.debug(f"[Parser] 🔍 Bloc interstage détecté (RX→CPU {rx_cpu} ms, CPU→GPU {cpu_gpu} ms, PROC→GPU {proc_gpu} ms, total={total} ms)")

        fid = self.last_frame_id or -1  # fallback si pas d'ID détecté
        return {
            "frame_id": fid,
            "event": "interstage",
            "latencies": {
                "rx_cpu": rx_cpu,
                "cpu_gpu": cpu_gpu,
                "proc_gpu": proc_gpu,
                "total": total,
            },
        }

    # ------------------------------------------------------------------ #
    def _extract_ts(self, line: str):
        """Extrait un timestamp log (format [YYYY-MM-DD HH:MM:SS,mmm])."""
        m = re.search(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]", line)
        if not m:
            return None
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
        return dt.timestamp() + float(m.group(3)) / 1000.0
