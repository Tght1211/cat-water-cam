"""低成本的全画面变化触发，不依赖猫类别、毛色、水碗位置或模型结论。"""
import cv2


class MotionDetector:
    def __init__(self, pixel_threshold=6, area_ratio=0.0005, reference_seconds=0.2):
        self.pixel_threshold = pixel_threshold
        self.area_ratio = area_ratio
        self.reference_seconds = reference_seconds
        self.reference = None
        self.reference_at = None

    def reset(self):
        self.reference = None
        self.reference_at = None

    def observe(self, now, frame):
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (320, max(1, round(height * 320 / width))))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if (self.reference is None or self.reference.shape != gray.shape
                or now - self.reference_at > 2):
            self.reference, self.reference_at = gray, now
            return False
        changed = cv2.absdiff(self.reference, gray) >= self.pixel_threshold
        moving = float(changed.mean()) >= self.area_ratio
        # 保留短时间基准，使缓慢移动的累计变化也能触发。
        if now - self.reference_at >= self.reference_seconds:
            self.reference, self.reference_at = gray, now
        return moving
