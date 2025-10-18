import json
import logging
from core.monitoring.kpi import format_kpi, parse_kpi_string, KpiJsonFormatter


def test_format_and_parse_kpi_roundtrip():
    data = {"ts": 12345.6789, "fps_in": "30", "latency_ms": "12.3"}
    s = format_kpi(data)
    parsed = parse_kpi_string(s)
    assert float(parsed.get("ts")) == float(data["ts"])
    assert parsed.get("fps_in") == data["fps_in"]


def test_kpi_json_formatter_outputs_json():
    logger = logging.getLogger("test.kpi.json")
    rec = logging.LogRecord(name=logger.name, level=logging.INFO, pathname=__file__, lineno=1, msg=format_kpi({"ts": 1.23, "event": "test"}), args=(), exc_info=None)
    fmt = KpiJsonFormatter()
    out = fmt.format(rec)
    obj = json.loads(out)
    assert obj.get("event") == "test"