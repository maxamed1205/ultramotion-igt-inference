import time
import numpy as np

from core.types import FrameMeta, Pose, IGTLinkContract


def test_frame_meta_conversion_roundtrip():
    fm = FrameMeta(frame_id=10, ts=time.time(), orientation="UN", coord_frame="Echographique")
    d = fm.to_igt_dict()
    assert d["DeviceName"] == "Image"
    assert len(d["Spacing"]) == 3
    assert isinstance(d["Timestamp"], float)


def test_pose_from_igt_matrix():
    class Msg:
        matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    p = Pose.from_igt(Msg())
    assert p.valid and p.matrix.shape == (4,4)


def test_igtlink_header_parse():
    b = b"Image".ljust(20, b"\x00") + b"IMAGE".ljust(12, b"\x00") + (12345).to_bytes(8, "big") + (2048).to_bytes(8, "big") + b"\x00"*8
    h = IGTLinkContract.from_bytes(b)
    assert h.device_name == "Image"
    assert h.message_type.strip() == "IMAGE"
