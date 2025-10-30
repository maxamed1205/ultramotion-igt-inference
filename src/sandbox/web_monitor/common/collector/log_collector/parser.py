"""
parser.py
----------
Analyse les lignes des logs (regex précompilées et dispatch rapide).
Renvoie un dictionnaire minimal pour agrégation.
"""

import re
from datetime import datetime
from . import logger


class LogParser:
    """Détecte et extrait les événements des logs pipeline.log et kpi.log."""

    def __init__(self):
        # ─── Regex précompilées (accès ultra-rapide)
        # Use case-insensitive matching for the simple frame-id patterns so
        # both "Frame #001" and "frame #001" are recognised in logs.
        self.patterns = {
            "rx": re.compile(r"\[DATASET-RX\].*frame #(\d+)", re.I),
            "proc": re.compile(r"\[PROC-SIM\].*frame #(\d+)", re.I),
            "tx": re.compile(r"\[TX-SIM\].*frame #(\d+)", re.I),
            "copy_async": re.compile(
                r"event=copy_async.*norm_ms=([\d.]+).*pin_ms=([\d.]+).*copy_ms=([\d.]+).*total_ms=([\d.]+).*frame=(\d+)"
            ),
            "interstage": re.compile(
                r"RX → CPU-to-GPU:\s*([\d.]+)ms.*?"
                r"CPU-to-GPU → PROC:\s*([\d.]+)ms.*?"
                r"PROC → GPU-to-CPU:\s*([\d.]+)ms.*?"
                r"Total processing:\s*([\d.]+)",
                re.S,
            ),
        }

        self.last_frame_id = None       # garde en mémoire le dernier frame_id
        self._pending_interstage = ""   # buffer multi-lignes

    # ------------------------------------------------------------------ #
    def parse_line(self, line: str, source: str):
        """Dispatch principal rapide (pipeline/kpi)."""
        line = line.strip()
        if not line:
            return None

        # ─────────────────────────────
        # 1️⃣ Détection bloc Inter-stage (multi-lignes)
        # ─────────────────────────────
        if "Inter-stage" in line:
            # début de bloc
            self._pending_interstage = line + "\n"
            logger.debug("[Parser] Début bloc interstage détecté")
            return None

        elif self._pending_interstage:
            # accumulation
            self._pending_interstage += line + "\n"
            if "Total processing" in line:
                # fin du bloc -> parse complet
                parsed = self._parse_interstage(self._pending_interstage)
                self._pending_interstage = ""
                if parsed:
                    logger.debug(f"[Parser] Bloc interstage complété pour frame #{parsed['frame_id']}")
                return parsed
            return None

        # ─────────────────────────────
        # 2️⃣ Événements classiques RX/PROC/TX/copy_async
        # ─────────────────────────────
        if "[DATASET-RX]" in line:
            return self._parse_event(line, "rx")
        elif "[PROC-SIM]" in line:
            return self._parse_event(line, "proc")
        elif "[TX-SIM]" in line:
            return self._parse_event(line, "tx")
        elif "event=copy_async" in line:
            return self._parse_copy_async(line)

        return None

    # ------------------------------------------------------------------ #
    def _parse_event(self, line: str, tag: str):
        """Parse RX/PROC/TX lines with timestamp."""
        m = self.patterns[tag].search(line)
        if not m:
            return None
        fid = int(m.group(1))
        self.last_frame_id = fid
        ts = self._extract_ts(line)
        logger.debug(f"[Parser] {tag.upper()} détecté frame #{fid}")
        return {"frame_id": fid, "event": tag, "ts": ts}

    # ------------------------------------------------------------------ #
    def _parse_copy_async(self, line: str):
        m = self.patterns["copy_async"].search(line)
        if not m:
            return None
        norm_ms, pin_ms, copy_ms, total_ms, fid = m.groups()
        fid = int(fid)
        self.last_frame_id = fid
        logger.debug(f"[Parser] copy_async détecté frame #{fid}")
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
        if not m:
            logger.debug("[Parser] Bloc interstage non reconnu")
            return None
        rx_cpu, cpu_gpu, proc_gpu, total = map(float, m.groups())
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
