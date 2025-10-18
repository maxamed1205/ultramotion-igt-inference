import os  # pour lire les variables d'environnement (activation/désactivation du KPI)
import logging  # système standard de logs
import threading  # verrouillage thread-safe pour les compteurs
from typing import Dict  # typage du dictionnaire
import json  # conversion vers JSON pour le formatter JSONL

LOG_KPI = logging.getLogger("igt.kpi")  # logger principal pour les lignes KPI
_LOG_FALLBACK = logging.getLogger("igt.monitor")  # logger secondaire (fallback) en cas d’échec d’écriture

# Schéma de message KPI par défaut (référence, pas forcément utilisé directement)
KPI_FORMAT = "ts={ts:.3f} fps_rx={fps_rx:.2f} fps_tx={fps_tx:.2f} latency_ms={latency_ms:.1f} drops={drops}"


def is_kpi_enabled() -> bool:  # vérifie si le logging KPI est activé (via variable d’environnement)
    return os.getenv("KPI_LOGGING", "1") not in ("0", "false", "False")  # désactivé seulement si KPI_LOGGING=0|false


def safe_log_kpi(msg: str, *args, **kwargs):  # écrit une ligne KPI dans le logger "igt.kpi"
    """Write a KPI line to the KPI logger if enabled.  # écrit une ligne KPI dans le logger dédié

    This is best-effort and must never raise to the main flow.  # ne doit jamais lever d’exception vers le flot principal
    """
    if not is_kpi_enabled():  # si le KPI est désactivé
        return  # ne rien faire
    try:
        LOG_KPI.info(msg, *args, **kwargs)  # envoie le message KPI au logger dédié (niveau INFO)
    except Exception:
        # KPI policy: never raise/logging errors into the main flow  # en cas d’erreur de logging, jamais interrompre le programme
        try:
            _LOG_FALLBACK.debug("Failed to write KPI: %s", msg)  # log de secours dans "igt.monitor"
        except Exception:
            pass  # ignore toute erreur ici aussi


def format_kpi(data: dict) -> str:  # formate un dictionnaire en chaîne "clé=valeur"
    """Build a stable KPI message string from a dict.  # construit une chaîne KPI stable à partir d’un dict

    Output example: "kpi ts=1697660000.123 fps_in=30 fps_out=29 latency_ms=12.3"  # exemple de sortie
    The caller should not include the leading `kpi` or ts; this helper will include ts.  # ts ajouté automatiquement
    Values will be converted to simple scalars and spaces are used as separators.  # chaque champ est séparé par un espace
    """
    if not isinstance(data, dict):  # validation du type d’entrée
        raise TypeError("format_kpi expects a dict")  # erreur si ce n’est pas un dict
    parts = []  # liste des paires clé=valeur
    import time  # pour générer le timestamp

    ts = float(data.get("ts", time.time()))  # récupère ts (ou maintenant si absent)
    parts.append(f"ts={ts:.6f}")  # ajoute la première clé "ts" formatée avec 6 décimales
    for k, v in data.items():  # itère sur les autres clés
        if k == "ts":  # ignore le timestamp (déjà ajouté)
            continue
        sval = str(v).replace(' ', '_').replace(',', '_').replace(';', '_')  # nettoie les caractères indésirables
        parts.append(f"{k}={sval}")  # ajoute la paire clé=valeur
    return " ".join(parts)  # joint toutes les paires avec des espaces


def parse_kpi_string(s: str) -> Dict[str, object]:  # parse une ligne KPI en dictionnaire Python
    """Parse a KPI message produced by format_kpi back into a dict.  # reconvertit une chaîne KPI en dict

    Example input: 'ts=12345.123456 fps_in=30 fps_out=29 latency_ms=12.3'  # exemple
    Returns a dict with 'ts' as float and other values as strings (best-effort).  # ts devient float, le reste reste string
    """
    out: Dict[str, object] = {}  # dict résultat
    if not s:  # si vide
        return out  # renvoie dict vide
    parts = s.strip().split()  # découpe la chaîne par espaces
    for p in parts:  # parcourt chaque segment
        if '=' not in p:  # si le segment ne contient pas de '='
            continue  # ignore
        k, v = p.split('=', 1)  # sépare la clé et la valeur
        if k == 'ts':  # cas spécial du timestamp
            try:
                out[k] = float(v)  # convertit en float
            except Exception:
                out[k] = v  # garde brut si échec
        else:
            out[k] = v  # stocke la valeur telle quelle
    return out  # retourne le dictionnaire résultant


class KpiJsonFormatter(logging.Formatter):  # formatter pour convertir un message KPI en JSONL
    """Formatter that converts a KPI record message into a JSON line.  # convertit une ligne KPI en JSON

    It expects the record message to be the output of format_kpi() (key=value pairs).  # le message d’entrée doit être au format clé=valeur
    """
    def format(self, record: logging.LogRecord) -> str:  # redéfinit la méthode format
        try:
            msg = record.getMessage()  # récupère le message brut du log
            data = parse_kpi_string(msg)  # le parse en dict
            data.setdefault("logger", record.name)  # ajoute le nom du logger s’il n’existe pas
            data.setdefault("level", record.levelname)  # ajoute le niveau du log s’il n’existe pas
            return json.dumps(data, ensure_ascii=False)  # retourne la ligne JSON encodée
        except Exception:
            # fallback to the raw message  # en cas d’erreur, on garde le message brut
            try:
                return json.dumps({"msg": record.getMessage(), "logger": record.name})  # JSON minimal
            except Exception:
                return record.getMessage()  # retourne le message brut s’il y a encore une erreur


# Persistent drop counters (thread-safe)  # compteurs persistants pour les frames perdues, protégés par verrou
_drop_lock = threading.Lock()  # verrou global pour protéger les écritures concurrentes
_drop_counters: Dict[str, int] = {}  # dictionnaire des compteurs de pertes (clé = nom, valeur = total)


def increment_drops(name: str = "acq.drop_total", delta: int = 1, emit: bool = True) -> int:  # incrémente un compteur de pertes
    """Increment a named drop counter and return the new value.  # incrémente un compteur nommé et renvoie la nouvelle valeur

    If `emit` is True, attempt to write the new total as a KPI line when KPI  # si emit=True, écrit aussi une ligne KPI
    logging is enabled. When KPI logging is disabled, the value is emitted to  # sinon log en DEBUG dans igt.monitor
    the monitoring logger at DEBUG level to preserve visibility without  # pour conserver la visibilité sans fichier KPI
    writing to the KPI file.
    """
    with _drop_lock:  # section critique (accès concurrent)
        val = _drop_counters.get(name, 0) + int(delta)  # récupère la valeur actuelle et ajoute le delta
        _drop_counters[name] = val  # met à jour le compteur

    # best-effort emission  # tentative d’émission sans bloquer ni lever d’exception
    try:
        if emit and is_kpi_enabled():  # si le KPI est activé et émission demandée
            import time  # pour le timestamp
            kmsg = format_kpi({"ts": time.time(), name: val})  # formate la ligne KPI avec la clé personnalisée
            safe_log_kpi(kmsg)  # écrit la ligne KPI
        elif emit:  # sinon si KPI désactivé
            _LOG_FALLBACK.debug("%s=%d", name, val)  # écrit en DEBUG dans igt.monitor
    except Exception:
        try:
            _LOG_FALLBACK.debug("Failed to emit drop counter %s", name)  # log fallback si erreur d’émission
        except Exception:
            pass  # ignore toute exception

    return val  # retourne la nouvelle valeur du compteur


def get_drop_count(name: str = "acq.drop_total") -> int:  # récupère la valeur actuelle d’un compteur
    with _drop_lock:  # section critique (thread-safe)
        return int(_drop_counters.get(name, 0))  # retourne la valeur (0 si absent)
