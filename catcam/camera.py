"""采集失败后的限速重连；相机句柄只由采集线程读写。"""
import time
import cv2


class ReconnectingCapture:
    def __init__(self,source,width,height,factory=cv2.VideoCapture,clock=time.monotonic,log=print,fps=30):
        self.source,self.width,self.height=source,width,height
        self.factory,self.clock,self.log=factory,clock,log
        self.fps=fps
        self.capture=None;self.failed_since=None;self.next_retry=0;self.delay=1
        self._open()

    def _open(self):
        if self.capture is not None:self.capture.release()
        self.capture=self.factory(self.source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
        self.capture.set(cv2.CAP_PROP_FPS,self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.next_retry=self.clock()+self.delay
        self.delay=min(10,self.delay*2)

    def isOpened(self):return self.capture is not None and self.capture.isOpened()

    def read(self):
        now=self.clock()
        if not self.isOpened():
            if now<self.next_retry:return False,None
            self._open()
        if not self.isOpened():return False,None
        ok,frame=self.capture.read()
        now=self.clock()
        if ok and frame is not None:
            if self.failed_since is not None:self.log('摄像头画面已恢复。')
            self.failed_since=None;self.delay=1
            return True,frame
        if self.failed_since is None:self.failed_since=now
        if now-self.failed_since>=3 and now>=self.next_retry:
            self.log('摄像头未返回新画面，正在重新连接。')
            self._open()
        return False,None

    def release(self):
        if self.capture is not None:self.capture.release();self.capture=None


class FreshMacCapture:
    """每次打开使用新的AVFoundation进程，避免拔插后的设备缓存与read阻塞。"""
    def __init__(self,source,width,height,executable='ffmpeg',fps=30):
        import subprocess
        self.width,self.height=width,height
        self.process=subprocess.Popen([
            executable,'-nostdin','-hide_banner','-loglevel','error',
            '-f','avfoundation','-framerate',str(fps),'-video_size','640x480','-i',f'{source}:none',
            '-an','-vf',f'scale={width}:{height},fps={fps}','-pix_fmt','bgr24',
            '-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
        self.buffer=bytearray()

    def set(self,*args):return True

    def isOpened(self):return self.process.poll() is None

    def read(self):
        import os
        import select
        import numpy as np
        size=self.width*self.height*3
        deadline=time.monotonic()+3
        while len(self.buffer)<size and self.isOpened():
            remaining=deadline-time.monotonic()
            if remaining<=0:return False,None
            ready,_,_=select.select([self.process.stdout],[],[],remaining)
            if not ready:return False,None
            chunk=os.read(self.process.stdout.fileno(),size-len(self.buffer))
            if not chunk:return False,None
            self.buffer.extend(chunk)
        if len(self.buffer)<size:return False,None
        frame=np.frombuffer(bytes(self.buffer[:size]),dtype=np.uint8).reshape(self.height,self.width,3)
        del self.buffer[:size]
        return True,frame

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except __import__('subprocess').TimeoutExpired:
                self.process.kill();self.process.wait(timeout=2)
        if self.process.stdout:self.process.stdout.close()


def mac_camera_name(index,executable='ffmpeg'):
    """由当前设备列表解析配置索引，只在首次设置设备名时使用。"""
    import re
    import subprocess
    result=subprocess.run([executable,'-hide_banner','-f','avfoundation','-list_devices','true','-i',''],
                          capture_output=True,text=True,timeout=8)
    video=result.stderr.split('AVFoundation video devices:')[-1].split('AVFoundation audio devices:')[0]
    for line in video.splitlines():
        match=re.search(r'\[(\d+)\]\s+(.+)$',line)
        if match and int(match.group(1))==index:return match.group(2).strip()
    return None
