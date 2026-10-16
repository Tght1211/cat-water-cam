"""无红外摄像头的弱光处理。

预览使用降噪、局部对比与自适应提亮；识别画质仍由原始帧判定。
固定背景的暗色前景变化可触发候选录像，不能直接证明舔水动作。
全黑、曝光变化和随机传感器噪声不作为猫到访证据。
"""
from __future__ import annotations

import cv2
import numpy as np


def assess_quality(frame, roi=(0, 0, 1, 1), minimum_brightness=12.0):
    """在增强之前检查原始饮水区域；全黑和无细节画面不作为有效监控。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    crop = frame[max(0, int(y1*h)):min(h, int(y2*h)), max(0, int(x1*w)):min(w, int(x2*w))]
    if crop.size == 0:
        crop = frame
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(np.percentile(gray, 90)-np.percentile(gray, 10))
    usable = brightness >= minimum_brightness and contrast >= 5
    return {'usable': usable, 'brightness': round(brightness, 1), 'contrast': round(contrast, 1),
            'status': 'unavailable' if not usable else ('low_light' if brightness < 50 else 'normal'),
            'message': ('光线或画面细节不足，暂停记录；请检查镜头并补光。' if not usable
                        else ('弱光监控，动作识别可靠性降低。' if brightness < 50 else '画面可用于监控。'))}

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def mean_brightness(frame) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def is_dark(frame, threshold: float = 50.0) -> bool:
    return mean_brightness(frame) < threshold


def enhance_lowlight(frame):
    """弱光增强：LAB 的 L 通道 CLAHE，拉出暗部轮廓而不整体过曝。返回 BGR。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab2 = cv2.merge((_clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def enhance_preview(frame):
    """实时预览：降噪、局部对比与自适应伽马；不生成缺失的场景细节。"""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    value=hsv[:,:,2]
    if np.percentile(value,99)<=2:
        return frame.copy()
    # 空间滤波不混合前后帧，避免猫嘴动作产生拖影。
    clean=cv2.bilateralFilter(frame,5,12,4)
    hsv=cv2.cvtColor(clean,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    v=cv2.createCLAHE(clipLimit=1.4,tileGridSize=(8,8)).apply(v)
    level=max(float(np.percentile(v,60)),1)/255
    gamma=float(np.clip(np.log(105/255)/np.log(min(level,.99)),.28,1))
    lut=np.round(255*(np.arange(256)/255)**gamma).astype(np.uint8)
    v=cv2.LUT(v,lut)
    # 暗部颜色噪声容易被放大，弱光时适当降低饱和度。
    s=np.round(s*(.25 if mean_brightness(frame)<20 else .65)).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)


class PreviewEnhancer:
    """静止区域做短时降噪，明显变化的像素直接使用当前帧。"""
    def __init__(self):
        self.previous=None

    def reset(self):
        self.previous=None

    def enhance(self,frame):
        current=frame.astype(np.float32)
        if self.previous is not None and self.previous.shape==current.shape:
            diff=np.max(np.abs(current-self.previous),axis=2)
            moving=cv2.dilate((diff>4).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
            current=np.where(moving[:,:,None],current,.65*self.previous+.35*current)
        self.previous=current
        return enhance_preview(np.round(current).astype(np.uint8))


class LowLightPresenceDetector:
    """固定机位的暗色前景候选；使用原始帧，不把轮廓当作舔水证据。"""
    def __init__(self):
        self.background=None
        self.started=None
        self.last=None
        self.roi=None

    def reset(self):
        self.background=None
        self.started=None
        self.last=None

    def observe(self,frame,roi,now):
        h,w=frame.shape[:2]
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(320,max(1,round(h*320/w))))
        gray=cv2.GaussianBlur(gray,(5,5),0).astype(np.float32)
        gh,gw=gray.shape
        # 向水盆上方扩展，覆盖凑过来的猫头。
        x1,y1,x2,y2=roi
        rect=(int(x1*gw),max(0,int((y1-(y2-y1)*.5)*gh)),int(x2*gw),int(y2*gh))
        if self.background is None or self.background.shape!=gray.shape or self.roi!=tuple(roi) or self.last is None or now-self.last>2:
            self.background=gray.copy();self.started=now;self.roi=tuple(roi)
        self.last=now
        if float(np.percentile(gray,90))<3:
            return {'ready':False,'present':False,'area_ratio':0.0}
        xa,ya,xb,yb=rect
        outside=np.ones(gray.shape,bool);outside[ya:yb,xa:xb]=False
        residual=self.background-gray
        outside_delta=residual[outside]
        exposure=float(np.median(outside_delta)) if outside_delta.size else 0
        noise=float(np.median(np.abs(outside_delta-exposure))) if outside_delta.size else 0
        threshold=max(2.0,noise*5)
        # 去掉整个画面的曝光变化，保留进入水盆上方的连通暗色前景。
        delta=residual-exposure
        mask=(delta[ya:yb,xa:xb]>threshold).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,_,components,_=cv2.connectedComponentsWithStats(mask)
        area=mask.size
        largest=max((c[cv2.CC_STAT_AREA] for c in components[1:]),default=0)
        ratio=largest/max(area,1)
        ready=now-self.started>=2
        present=ready and ratio>=.06
        # 前景保持原背景，猫停住后也不会立即消失；仅缓慢更新空场区域。
        frozen=cv2.dilate((np.abs(delta)>threshold).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        adjusted=gray+exposure
        self.background=np.where(frozen,self.background,.985*self.background+.015*adjusted)
        return {'ready':ready,'present':bool(present),'area_ratio':round(float(ratio),3)}
