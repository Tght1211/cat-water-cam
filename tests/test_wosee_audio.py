import numpy as np
import pytest
from catcam.wosee_audio import WoSeeAudioRing


def test_camera_pcm_uses_its_own_clock_and_pads_late_audio():
    ring = WoSeeAudioRing('wosee://127.0.0.1:18765/test', max_seconds=2)
    ring._receive_chunk(101, b'\x01\x02'*16000)
    assert ring.status()['available']
    result = ring.slice(99.5, 100.5)
    assert result == b'\0\0'*8000+b'\x01\x02'*8000
    assert ring.slice(90, 91) is None


def test_camera_audio_is_bounded_and_rejects_invalid_packets():
    ring = WoSeeAudioRing('wosee://127.0.0.1:18765/test', max_seconds=1)
    ring._receive_chunk(101, b'aa'*16000)
    ring._receive_chunk(102, b'bb'*16000)
    assert ring.status()['buffered_seconds'] == 1
    assert ring.slice(101, 102) == b'bb'*16000
    with pytest.raises(ValueError):ring._receive_chunk(103, b'odd')
    with pytest.raises(ValueError):ring._receive_chunk(103, 'error')


def test_camera_connector_checks_device_and_keeps_token_out_of_url():
    ring = WoSeeAudioRing('wosee://127.0.0.1:18765/test')
    ring.bridge._json = lambda path: {'serial':'test'} if path=='/api/status' else {'token':'private-token'}
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def recv(self,timeout):
            ring._stopped.set()
            return b'ab'*320
    def connector(url, **kwargs):
        assert url == 'ws://127.0.0.1:18765/api/audio/listen'
        assert kwargs['proxy'] is None
        assert kwargs['subprotocols']==['wosee-audio','private-token']
        return Socket()
    ring.connector=connector
    ring._reader()
    assert ring.available
    assert 'private-token' not in str(ring.status())
