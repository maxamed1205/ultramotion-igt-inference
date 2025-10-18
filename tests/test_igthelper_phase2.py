from service.igthelper import IGTGateway
import time


def test_gateway_start_stop():
    gw = IGTGateway("127.0.0.1", 18944, 18945)
    gw.start()
    assert gw.is_running
    time.sleep(2)
    status = gw.get_status()
    assert "fps_rx" in status
    gw.stop()
    assert not gw.is_running
