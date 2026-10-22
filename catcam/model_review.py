"""新版本重新评估已录候选，保留人工结论；进度可查询。"""
import threading
from catcam.judge import is_extreme_verdict


class ModelReview:
    def __init__(self,runtime,stats,feedback,recorder,auto_confirm_enabled=lambda:False):
        self.runtime,self.stats,self.feedback,self.recorder=runtime,stats,feedback,recorder
        self._lock=threading.Lock()
        self._status={'state':'idle','done':0,'total':0,'model':None}
        self.auto_confirm_enabled=auto_confirm_enabled

    def status(self):
        with self._lock:return dict(self._status)

    def start(self):
        judge,mode=self.runtime.snapshot()
        if judge is None:return False
        with self._lock:
            if self._status['state']=='running':return False
            self._status={'state':'running','done':0,'total':0,'model':judge.version}
        threading.Thread(target=self._run,args=(judge,mode),daemon=True).start()
        return True

    def _run(self,judge,mode):
        try:
            clips=self.recorder.list_clips()
            candidates=self.stats.review_candidates()
            with self._lock:self._status['total']=len(clips)
            for i,path in enumerate(clips):
                current,_=self.runtime.snapshot()
                if current is None or current.version!=judge.version:
                    with self._lock:self._status['state']='cancelled'
                    if current is not None:
                        self.start()
                    return
                if not candidates.get(path.name,True) or self.stats.requires_visual_review(path.name):
                    with self._lock:self._status['done']=i+1
                    continue
                result=judge.judge(path)
                self.feedback.record_video_duration(path)
                if result:
                    self.stats.set_prediction(path.name,int(result.drinking),judge.version,result.confidence)
                    if (mode=='gate' and not result.uncertain or self.auto_confirm_enabled() and is_extreme_verdict(result)):
                        self.feedback.record_machine_label(path.name,result.drinking,source='local',confidence=result.confidence)
                with self._lock:self._status['done']=i+1
            with self._lock:self._status['state']='done'
        except Exception as e:
            with self._lock:self._status.update(state='error',detail=str(e))
