import sys
import os
import traceback
# ensure 'src' is on path for running as script from repo root
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import importlib.util

# load detection_and_engine as a standalone module to avoid package import side-effects
mod_path = os.path.join(src_path, "core", "inference", "detection_and_engine.py")
spec = importlib.util.spec_from_file_location("detection_and_engine", mod_path)
dmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmod)
_is_mobilesam = dmod._is_mobilesam
print('Running MobileSAM detection sanity check')
# check on a simple dummy object
class Dummy:
    pass
print('Dummy ->', _is_mobilesam(Dummy()))

# try to import and build mobilesam model (best-effort)
try:
    from core.inference.MobileSAM.mobilesam_loader import build_mobilesam_model
    try:
        m = build_mobilesam_model(checkpoint=None)
        print('build_mobilesam_model() returned:', type(m))
        print('is mobilesam ->', _is_mobilesam(m))
    except Exception as e:
        print('build_mobilesam_model() failed:', e)
        traceback.print_exc()
except Exception as e:
    print('could not import build_mobilesam_model:', e)
    traceback.print_exc()

print('Done')
