"""普通变化录像进入回放；发现猫或人工指定后才进入复核。"""
import cv2


class ReviewRouter:
    def __init__(self, detector, stats, feedback):
        self.detector, self.stats, self.feedback = detector, stats, feedback

    def route(self, path):
        meta = self.feedback.label_meta(path.name)
        if meta and meta.get('source') == 'human':
            self.stats.set_review_candidate(path.name, True)
            return True
        capture = cv2.VideoCapture(str(path))
        candidate = False
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # 覆盖整段的多个位置，原视频仍在回放中保留，不依赖喝水分类器得分。
            positions = sorted({round(i * max(0,count-1) / 11) for i in range(12)})
            for index in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok and self.detector.detect_cats(frame):
                    candidate = True
                    break
        finally:
            capture.release()
        # 后台扫描期间，用户可能已经手动把这段加入复核，不能撤回用户的选择。
        if self.stats.review_candidates().get(path.name) is True:
            candidate = True
        self.stats.set_review_candidate(path.name, candidate)
        return candidate
