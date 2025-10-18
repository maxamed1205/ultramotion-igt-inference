import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
SMOKE = ROOT / 'src' / 'scripts' / 'smoke_logging_test.py'


def run_smoke(async_on: bool, run_id: str):
    env = os.environ.copy()
    env['LOG_MODE'] = 'dev'
    env['ASYNC_LOG'] = '1' if async_on else '0'
    env['RUN_ID'] = run_id
    os.makedirs(LOGS, exist_ok=True)
    for f in ('pipeline.log', 'error.log', 'kpi.log'):
        p = LOGS / f
        if p.exists():
            p.unlink()
    r = subprocess.run(['python', str(SMOKE)], env=env, cwd=str(ROOT))
    return r.returncode


def test_pipeline_has_no_error_entries():
    rc = run_smoke(False, 'sync_audit')
    assert rc == 0
    time.sleep(0.2)
    rc2 = run_smoke(True, 'async_audit')
    assert rc2 == 0

    pl_text = (LOGS / 'pipeline.log').read_text(encoding='utf-8') if (LOGS / 'pipeline.log').exists() else ''
    # Assert pipeline contains no "ERROR" level lines (they must be in error.log only)
    assert 'ERROR' not in pl_text
