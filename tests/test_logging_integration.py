import os
import subprocess
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
SMOKE = ROOT / 'src' / 'scripts' / 'smoke_logging_test.py'

KPI_RE = re.compile(r'^kpi ts=(?P<ts>\d+\.\d+) (?P<pairs>.+)$')


def run_smoke(async_on: bool, run_id: str) -> int:
    env = os.environ.copy()
    env['LOG_MODE'] = 'dev'
    env['ASYNC_LOG'] = '1' if async_on else '0'
    env['RUN_ID'] = run_id
    # ensure fresh logs
    os.makedirs(LOGS, exist_ok=True)
    for f in ('pipeline.log', 'error.log', 'kpi.log'):
        p = LOGS / f
        if p.exists():
            p.unlink()
    r = subprocess.run(['python', str(SMOKE)], env=env, cwd=str(ROOT))
    return r.returncode


def parse_kpi_lines():
    kfile = LOGS / 'kpi.log'
    if not kfile.exists():
        return []
    out = []
    with kfile.open('r', encoding='utf-8') as f:
        for line in f:
            m = KPI_RE.search(line)
            if m:
                out.append(m.group('pairs'))
    return out


def test_logging_no_dup_and_kpi_parse():
    # run sync
    rc = run_smoke(False, 'sync_pytest')
    assert rc == 0
    time.sleep(0.2)

    # run async
    rc2 = run_smoke(True, 'async_pytest')
    assert rc2 == 0

    # Ensure files exist
    assert (LOGS / 'pipeline.log').exists()
    assert (LOGS / 'error.log').exists()
    assert (LOGS / 'kpi.log').exists()

    pl = (LOGS / 'pipeline.log').read_text(encoding='utf-8')
    errors = (LOGS / 'error.log').read_text(encoding='utf-8')

    # pipeline.log should NOT contain error entries that are only in error.log
    assert 'SMOKE ERROR [sync_pytest]' not in pl
    assert 'SMOKE ERROR [async_pytest]' not in pl

    # parse KPI lines
    kpis = parse_kpi_lines()
    assert len(kpis) >= 1
