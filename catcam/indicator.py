"""Optional user-framed red LED: steady warning versus repeated blinking."""
from collections import deque
from pathlib import Path
import json
import threading
import time
import cv2
import numpy as np


class IndicatorMonitor:
    def __init__(self,directory):
        self.path=Path(directory)/'indicator.json'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.roi=json.loads(self.path.read_text())['roi'] if self.path.exists() else None
        self.samples=deque()
        self.lock=threading.RLock()
        self.last_seen=0
        self.state='unconfigured'

    def set_roi(self,roi):
        try: roi=tuple(float(v) for v in roi)
        except (TypeError,ValueError): raise ValueError('请框选指示灯区域')
        if len(roi)!=4 or not all(0<=v<=1 for v in roi) or min(roi[2]-roi[0],roi[3]-roi[1])<.01:
            raise ValueError('指示灯区域无效')
        with self.lock:
            self.roi=roi
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'roi':roi}))
            temp.replace(self.path)
            self.samples.clear()
            self.state='learning'
            return self.status()

    def observe(self,now,frame,usable,occluded=False,enabled=False):
        with self.lock:
            if not enabled or self.roi is None:
                self.state='unconfigured'
                return
            if now-self.last_seen<.25: return
            if now-self.last_seen>2: self.samples.clear()
            self.last_seen=now
            if not usable or occluded:
                self.samples.clear()
                self.state='unknown'
                return
            h,w=frame.shape[:2]
            x1,y1,x2,y2=self.roi
            crop=frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
            if crop.size==0:
                self.state='unknown'
                return
            hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            red=((hsv[:,:,0]<10)|(hsv[:,:,0]>170))&(hsv[:,:,1]>120)&(hsv[:,:,2]>70)
            on=bool(red.mean()>.02 and red.sum()>=4)
            self.samples.append((now,on))
            while self.samples and now-self.samples[0][0]>12: self.samples.popleft()
            if not self.samples or now-self.samples[0][0]<8:
                self.state='learning'
                return
            values=[v for _,v in self.samples]
            ratio=sum(values)/len(values)
            switches=sum(a!=b for a,b in zip(values,values[1:]))
            self.state='blinking_red' if switches>=4 and .1<ratio<.9 else 'steady_red' if ratio>=.9 else 'off' if ratio<=.1 else 'unknown'

    def status(self,now=None):
        with self.lock:
            now=time.time() if now is None else now
            state='unknown' if self.last_seen and now-self.last_seen>3 else self.state
            return {'state':state,'roi':self.roi,'last_observed_at':self.last_seen}
