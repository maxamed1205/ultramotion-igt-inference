import os, sys
import numpy as np

# Ensure project src is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from core.inference.engine import orchestrator

# Monkeypatch the heavy functions with lightweight stubs

def fake_run_detection(dfine_model, frame):
    # Return a bbox within a 100x100 image and a high confidence
    return (10, 15, 60, 80), 0.9


def fake_run_segmentation(sam_model, roi):
    # roi is HxW or HxWxC; return a simple mask of the ROI shape
    import numpy as np
    h, w = roi.shape[0], roi.shape[1]
    m = np.zeros((h, w), dtype=np.uint8)
    # mark a central rectangle
    m[h//4: h*3//4, w//4: w*3//4] = 1
    return m


# Replace the module-level functions
orchestrator.run_detection = fake_run_detection
orchestrator.run_segmentation = fake_run_segmentation

# Create a dummy image 100x100 RGB
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

res = orchestrator.prepare_inference_inputs(img, dfine_model=None, sam_model=None, tau_conf=0.5)
print("Result keys:", sorted(res.keys()))
print("state_hint:", res.get('state_hint'))
print("bbox:", res.get('bbox'))
print("conf:", res.get('conf'))
print("mask shape:", None if res.get('mask') is None else res.get('mask').shape)
print("weights:", None if res.get('weights') is None else tuple(w.shape for w in res.get('weights')))

# Assert basic contract
assert isinstance(res, dict)
assert 'state_hint' in res
assert res['state_hint'] == 'VISIBLE'
assert res['bbox'] is not None
assert res['mask'] is not None
print('Quick test passed')
