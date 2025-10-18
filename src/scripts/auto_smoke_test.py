import os
import subprocess
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
SMOKE = ROOT / 'scripts' / 'smoke_logging_test.py'

def run_smoke(async_on: bool, run_id: str):
    env = os.environ.copy()
    env['LOG_MODE'] = 'dev'
    env['ASYNC_LOG'] = '1' if async_on else '0'
    env['RUN_ID'] = run_id
    # ensure fresh logs
    for f in ('pipeline.log', 'error.log', 'kpi.log'):
        p = LOGS / f
        if p.exists():
            p.unlink()
    print('Running smoke, ASYNC_LOG=', env['ASYNC_LOG'])
    r = subprocess.run(['python', str(SMOKE)], env=env, cwd=str(ROOT))
    return r.returncode

KPI_RE = re.compile(r'^kpi ts=(?P<ts>\d+\.\d+) (?P<pairs>.+)$')

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

if __name__ == '__main__':
    os.makedirs(LOGS, exist_ok=True)
    run_smoke(False, 'sync1')
    time.sleep(0.2)
    run_smoke(True, 'async1')

    # Check duplication: pipeline.log should not contain duplicate run ids
    pl = (LOGS / 'pipeline.log').read_text(encoding='utf-8') if (LOGS / 'pipeline.log').exists() else ''
    errors = (LOGS / 'error.log').read_text(encoding='utf-8') if (LOGS / 'error.log').exists() else ''

    dup_sync = 'SMOKE ERROR [sync1]' in pl and 'SMOKE ERROR [sync1]' in errors
    dup_async = 'SMOKE ERROR [async1]' in pl and 'SMOKE ERROR [async1]' in errors

    print('dup_sync (should be False):', dup_sync)
    print('dup_async (should be False):', dup_async)

    kpis = parse_kpi_lines()
    print('kpi lines parsed:', len(kpis))
    print('sample kpi:', kpis[:5])
