import numpy as np
from catcam.camera import ReconnectingCapture


def test_camera_reopens_after_sustained_read_failure_and_recovers():
    now=[0];opened=[]
    class Capture:
        def __init__(self,source):self.index=len(opened);opened.append(self);self.released=False
        def set(self,*args):pass
        def isOpened(self):return not self.released
        def read(self):return (True,np.zeros((2,2,3),np.uint8)) if self.index else (False,None)
        def release(self):self.released=True
    source=ReconnectingCapture(0,640,480,factory=Capture,clock=lambda:now[0],log=lambda s:None)
    assert not source.read()[0]
    now[0]=2;assert not source.read()[0];assert len(opened)==1
    now[0]=4;assert not source.read()[0];assert len(opened)==2 and opened[0].released
    assert source.read()[0]


def test_unavailable_camera_retries_with_backoff():
    now=[0];opened=[]
    class Capture:
        def __init__(self,source):opened.append(self)
        def set(self,*args):pass
        def isOpened(self):return False
        def release(self):pass
    source=ReconnectingCapture(0,640,480,factory=Capture,clock=lambda:now[0])
    for _ in range(10):assert not source.read()[0]
    assert len(opened)==1
    now[0]=1;source.read();assert len(opened)==2
    now[0]=2;source.read();assert len(opened)==2


def test_mac_camera_identity_is_resolved_from_video_list(monkeypatch):
    from types import SimpleNamespace
    from catcam.camera import mac_camera_name
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(stderr='AVFoundation video devices:\n[AVFoundation indev @ ptr] [0] UGREEN Camera\n[AVFoundation indev @ ptr] [1] MacBook Air相机\nAVFoundation audio devices:\n[AVFoundation indev @ ptr] [0] microphone'))
    assert mac_camera_name(0)=='UGREEN Camera'
    assert mac_camera_name(1)=='MacBook Air相机'
    assert mac_camera_name(3) is None
