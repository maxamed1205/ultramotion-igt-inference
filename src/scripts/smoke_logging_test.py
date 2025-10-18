import os
import yaml
import logging
import logging.config
import sys

# smoke test for logging configuration
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# ensure project `src/` (this directory) is on sys.path so package imports in logging.yaml resolve
SRC_PATH = ROOT
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
LOG_DIR = os.path.join(ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

cfg_path = os.path.join(ROOT, 'config', 'logging.yaml')
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# simulate LOG_MODE=dev
os.environ['LOG_MODE'] = 'dev'
# simulate enabling async logging for this smoke run
os.environ['ASYNC_LOG'] = '1'
log_mode = os.environ.get('LOG_MODE', 'perf').lower()
if 'handlers' in cfg and 'console' in cfg['handlers']:
    if log_mode == 'dev':
        cfg['handlers']['console']['level'] = 'INFO'
    else:
        cfg['handlers']['console']['level'] = 'WARNING'

logging.config.dictConfig(cfg)
# If ASYNC_LOG requested, initialize the async logging subsystem similar to main.py
if os.environ.get('ASYNC_LOG', '0') in ('1', 'true', 'on'):
    try:
        from core.monitoring.async_logging import setup_async_logging

        try:
            log_queue, listener = setup_async_logging(log_dir=LOG_DIR, attach_to_logger='igt', yaml_cfg=cfg, remove_yaml_file_handlers=True)
        except Exception:
            # fallback for older return style
            listener = setup_async_logging(log_dir=LOG_DIR, attach_to_logger='igt', yaml_cfg=cfg, remove_yaml_file_handlers=True)
        print('async listener started')
    except Exception as e:
        print('failed to start async listener ->', e)
        listener = None
else:
    listener = None
logger = logging.getLogger('igt.smoketest')
run_id = os.environ.get('RUN_ID', 'manual')
logger.info('SMOKE INFO: should appear on console in dev mode')
logger.error('SMOKE ERROR: should appear on console and in logs')
logger.info('SMOKE INFO [%s]: should appear on console in dev mode', run_id)
logger.error('SMOKE ERROR [%s]: should appear on console and in logs', run_id)
try:
    raise RuntimeError('smoke-test-exception')
except Exception:
    logger.exception('SMOKE EXC [%s]: captured exception', run_id)

print('--- check handlers for logger igt ---')
root_logger = logging.getLogger('igt')
for i, h in enumerate(root_logger.handlers):
    info = {
        'index': i,
        'type': type(h).__name__,
        'level': logging.getLevelName(h.level),
        'formatter': type(h.formatter).__name__ if h.formatter else None,
    }
    # some handlers (File handlers) have baseFilename attribute
    if hasattr(h, 'baseFilename'):
        info['baseFilename'] = getattr(h, 'baseFilename')
    print(info)

# Force a direct emit on file handlers to check lazy-open behavior
print('\n--- forcing direct emit on file handlers ---')
for h in root_logger.handlers:
    if hasattr(h, 'baseFilename'):
        try:
            record = logging.LogRecord(name='igt.forced', level=logging.ERROR, pathname=__file__, lineno=0, msg='FORCED WRITE', args=(), exc_info=None)
            h.acquire()
            try:
                h.emit(record)
            finally:
                h.release()
            print('wrote to', getattr(h, 'baseFilename'))
        except Exception as e:
            print('failed to write to handler', getattr(h, 'baseFilename', None), '->', e)

print('\n--- handler stream diagnostics ---')
file_paths = []
for h in root_logger.handlers:
    if hasattr(h, 'baseFilename'):
        stream = getattr(h, 'stream', None)
        info = {'baseFilename': h.baseFilename, 'stream': repr(stream), 'stream_closed': getattr(stream, 'closed', None)}
        print(info)
        file_paths.append(h.baseFilename)

print('\ncwd:', os.getcwd())
# list project-root/logs if exists (some handlers may point there)
checked_dirs = set([os.path.dirname(p) for p in file_paths])
for d in checked_dirs:
    print('listing', d, '->', os.path.exists(d) and os.listdir(d) or '(missing)')

print('--- check files')
print('logs dir exists:', os.path.exists(LOG_DIR))
print('error.log exists:', os.path.exists(os.path.join(LOG_DIR, 'error.log')))
print('pipeline.log exists:', os.path.exists(os.path.join(LOG_DIR, 'pipeline.log')))
print('kpi.log exists:', os.path.exists(os.path.join(LOG_DIR, 'kpi.log')))

# print last few lines of error.log and pipeline.log for quick verification
for fname in ['error.log', 'pipeline.log']:
    path = os.path.join(LOG_DIR, fname)
    print(f'--- {fname} ---')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            to_read = min(size, 4096)
            f.seek(max(0, size - to_read))
            tail = f.read().decode('utf-8', errors='replace')
            print(tail)
    else:
        print('(missing)')

print('SMOKE TEST DONE')
