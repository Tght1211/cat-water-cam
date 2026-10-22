"""Codex SDK 视频预标注：分窗看连续帧，只产草稿，不覆盖人工轨道或启动训练。"""
from __future__ import annotations
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import cv2
from catcam.database import connect
from catcam.drink_intervals import normalize_intervals
from catcam.ai_budget import AIBudget
def clip_duration(path):
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps>0 else 0
    finally:cap.release()


SCHEMA={'type':'object','additionalProperties':False,'required':['segments','unknown','reason'],
        'properties':{'segments':{'type':'array','items':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}},
                      'unknown':{'type':'array','items':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}},
                      'reason':{'type':'string'}}}


def validate_proposal(result,start,end):
    if not isinstance(result,dict) or not isinstance(result.get('reason'),str) or len(result['reason'])>3000:
        raise ValueError('Agent结果格式无效')
    output={}
    for key in ('segments','unknown'):
        pairs=result.get(key)
        if not isinstance(pairs,list) or len(pairs)>200:raise ValueError('Agent区间过多或无效')
        pairs=normalize_intervals(pairs,end) if pairs else []
        if any(a<start for a,b in pairs):raise ValueError('Agent返回了窗口之外的区间')
        output[key]=pairs
    output['reason']=result['reason']
    return output


_sdk_processes=set()
_sdk_process_lock=threading.Lock()


def shutdown_sdk_calls():
    with _sdk_process_lock:
        for process in list(_sdk_processes):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass


def sdk_call(request,timeout=240):
    process=subprocess.Popen([sys.executable,'-m','catcam.codex_worker'],stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,start_new_session=True)
    with _sdk_process_lock:_sdk_processes.add(process)
    try:
        try:
            stdout,_=process.communicate(json.dumps(request),timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.communicate(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.communicate()
            raise ValueError('Codex 分析超时，请缩短视频或检查第三方服务')
        if len(stdout)>1024*1024:raise ValueError('Codex返回结果过大')
        try:result=json.loads(stdout)
        except ValueError:raise ValueError('Codex SDK 未返回结构化结果')
        if process.returncode or not result.get('ok'):raise ValueError(result.get('error','Codex SDK 调用失败'))
        return result['result']
    finally:
        with _sdk_process_lock:_sdk_processes.discard(process)


class CodexVideoAgent:
    def __init__(self,cfg,config_path,feedback,clips_dir,runner=sdk_call):
        self.cfg,self.config_path,self.feedback,self.clips_dir=cfg,Path(config_path),feedback,Path(clips_dir)
        self.runner=runner;self.lock=threading.Lock();self.running=False;self.stopping=threading.Event()
        with connect(feedback.db_path) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS codex_video_jobs (id TEXT PRIMARY KEY,clip TEXT,created REAL,status TEXT,progress INTEGER,total INTEGER,result TEXT,error TEXT)')
            connection.execute("UPDATE codex_video_jobs SET status='interrupted',error='服务重启，可重新提交' WHERE status IN ('queued','running')")

    def close(self):
        self.stopping.set()
        shutdown_sdk_calls()

    def settings(self):
        import importlib.util
        installed=importlib.util.find_spec('openai_codex') is not None
        return {'enabled':True,'sdk_installed':installed,'base_url':self.cfg.codex_base_url,
                'model':self.cfg.codex_model,'has_key':bool(self.cfg.codex_api_key),
                'daily_budget':self.cfg.codex_daily_budget,'max_video_seconds':self.cfg.codex_max_video_seconds,
                'protocol':'Responses API','running':self.running}

    def configure(self,body):
        base=body.get('base_url',self.cfg.codex_base_url).strip().rstrip('/')
        parsed=urlsplit(base)
        try:parsed.port
        except ValueError:raise ValueError('服务地址端口无效')
        if len(base)>2048 or parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('请填写有效的 Base URL（例如 https://服务地址/v1）')
        model=body.get('model',self.cfg.codex_model)
        key=body.get('api_key') or self.cfg.codex_api_key
        if not isinstance(model,str) or not model.strip() or len(model)>256 or not isinstance(key,str) or not key.strip() or len(key)>16384:
            raise ValueError('请填写模型名和 API Key')
        budget=body.get('daily_budget',self.cfg.codex_daily_budget)
        if type(budget) is not int or not 1<=budget<=1000:raise ValueError('每日窗口调用上限须为 1 至 1000')
        with self.lock:
            if self.running:raise ValueError('分析进行中，请完成后修改设置')
            self.cfg.codex_base_url=base;self.cfg.codex_model=model.strip();self.cfg.codex_api_key=key.strip();self.cfg.codex_daily_budget=budget
            temp=self.config_path.with_suffix('.tmp');temp.write_text(json.dumps(asdict(self.cfg),ensure_ascii=False,indent=2));temp.chmod(0o600);temp.replace(self.config_path)
        return self.settings()

    def test(self):
        if not self.cfg.codex_api_key:raise ValueError('请先保存 Agent 服务配置')
        budget=AIBudget(self.feedback.db_path,self.cfg.codex_daily_budget,0)
        request_id=budget.reserve(self.cfg.codex_model,'codex:synthetic_test')
        if request_id is None:raise ValueError('今日调用预算不足')
        try:
            import numpy as np
            with tempfile.TemporaryDirectory(prefix='catcam-sdk-test-') as directory:
                frame=np.full((160,240,3),255,np.uint8)
                cv2.rectangle(frame,(20,30),(100,110),(0,0,255),-1)
                cv2.circle(frame,(175,75),35,(255,0,0),-1)
                image=Path(directory)/'synthetic.jpg';cv2.imwrite(str(image),frame)
                result=self.runner({'directory':directory,'base_url':self.cfg.codex_base_url,'key':self.cfg.codex_api_key,
                    'model':self.cfg.codex_model,'schema':SCHEMA,'frames':[[0,str(image)]],
                    'prompt':'这是连接与视觉能力测试。请在 reason 中描述图片中的颜色和形状，segments和unknown都返回空数组。'})
                validate_proposal(result,0,1)
            budget.finish(request_id,'ok')
            return {'ok':True,'message':'SDK / Responses 已返回视觉描述：'+result['reason']}
        except Exception:
            budget.finish(request_id,'failed')
            raise ValueError('Codex SDK 测试失败：请检查第三方 Responses API、模型图像输入能力、密钥和服务额度')

    def status(self,ident=None):
        with connect(self.feedback.db_path) as connection:
            if ident:rows=connection.execute('SELECT * FROM codex_video_jobs WHERE id=?',(ident,)).fetchall()
            else:rows=connection.execute('SELECT * FROM codex_video_jobs ORDER BY created DESC LIMIT 50').fetchall()
        jobs=[{'id':r[0],'clip':r[1],'created':r[2],'status':r[3],'progress':r[4],'total':r[5],
               'result':json.loads(r[6]) if r[6] else None,'error':r[7]} for r in rows]
        return {'jobs':jobs,**self.settings()}

    def start(self,names):
        if self.stopping.is_set():raise ValueError('Agent 正在关闭')
        if not self.settings()['sdk_installed']:raise ValueError('请先安装 requirements.txt 中的官方 openai-codex SDK')
        if not self.cfg.codex_api_key or not self.cfg.codex_model or not self.cfg.codex_base_url:raise ValueError('请先配置 Codex Agent 的服务地址、模型和密钥')
        if not isinstance(names,list) or not 1<=len(names)<=10 or len(set(names))!=len(names):raise ValueError('每批请选择 1 至 10 段不同视频')
        planned=[]
        for name in names:
            if not isinstance(name,str) or any(s in name for s in ('/','\\','..')):raise ValueError('录像名称无效')
            path=self.clips_dir/name
            if not path.is_file() or path.is_symlink():raise ValueError('录像不存在')
            duration=clip_duration(path)
            if not 0<duration<=self.cfg.codex_max_video_seconds:raise ValueError(f'单段视频须在 {self.cfg.codex_max_video_seconds} 秒以内')
            planned.append((uuid.uuid4().hex,name,duration))
        budget=AIBudget(self.feedback.db_path,self.cfg.codex_daily_budget,0)
        if sum(math.ceil(d/30) for _,_,d in planned)>budget.status()['remaining']:raise ValueError('剩余调用预算不足；每 30 秒录像占用一次 Agent 窗口预算')
        with self.lock:
            if self.running:raise ValueError('已有一批分析任务进行中')
            self.running=True
            with connect(self.feedback.db_path) as connection:
                for ident,name,duration in planned:connection.execute('INSERT INTO codex_video_jobs VALUES(?,?,?,?,?,?,NULL,NULL)',(ident,name,time.time(),'queued',0,math.ceil(duration/30)))
            threading.Thread(target=self._run_batch,args=(planned,),daemon=True).start()
        return {'jobs':[p[0] for p in planned]}

    def _update(self,ident,**changes):
        allowed={'status','progress','result','error'}
        if not changes.keys()<=allowed:raise ValueError('任务字段无效')
        with connect(self.feedback.db_path) as connection:
            connection.execute('UPDATE codex_video_jobs SET '+','.join(k+'=?' for k in changes)+' WHERE id=?',(*changes.values(),ident))

    def _run_batch(self,planned):
        try:
            for ident,name,duration in planned:
                try:self._analyze(ident,name,duration)
                except Exception as error:
                    if self.stopping.is_set():
                        self._update(ident,status='interrupted',error='服务停止，分析已中断');continue
                    message=str(error) if isinstance(error,ValueError) else '分析失败；请检查 SDK、第三方服务及视频完整性'
                    self._update(ident,status='failed',error=message[:500])
        finally:
            with self.lock:self.running=False

    def _analyze(self,ident,name,duration):
        self._update(ident,status='running')
        segments=[];unknown=[];reasons=[]
        budget=AIBudget(self.feedback.db_path,self.cfg.codex_daily_budget,0)
        with tempfile.TemporaryDirectory(prefix='catcam-codex-') as tmp:
            for window in range(math.ceil(duration/30)):
                if self.stopping.is_set():raise ValueError('任务已中断')
                start,end=window*30,min(duration,(window+1)*30)
                directory=Path(tmp)/str(window);directory.mkdir()
                cap=cv2.VideoCapture(str(self.clips_dir/name));frames=[]
                try:
                    for index in range(max(1,math.ceil((end-start)*2))):
                        timestamp=min(end-.01,start+index*.5)
                        cap.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000);ok,frame=cap.read()
                        if not ok:raise ValueError('视频抽帧失败，本段未生成标注')
                        h,w=frame.shape[:2];frame=cv2.resize(frame,(max(1,round(w*640/max(h,w))),max(1,round(h*640/max(h,w)))))
                        path=directory/f'{index:03}.jpg'
                        if not cv2.imwrite(str(path),frame):raise ValueError('无法保存分析抽帧')
                        frames.append([timestamp,str(path)])
                finally:cap.release()
                request_id=budget.reserve(self.cfg.codex_model,'codex:'+name)
                if request_id is None:raise ValueError('调用预算不足，停止分析')
                try:
                    prompt=f'分析猫咪喝水动作。连续帧为同一视频的 {start:.3f} 至 {end:.3f} 秒，每0.5秒一帧。只把实际持续舔水列为segments；靠近、嗅闻、抬头、离开不算；遮挡或看不清列为unknown。返回原视频绝对秒数的多段[start,end]，范围必须在该窗口内。无动作返回空数组。不要根据饮水机是否出水推断猫是否舔水，不要编造精确帧级时刻。reason简短中文说明。'
                    result=validate_proposal(self.runner({'directory':str(directory),'base_url':self.cfg.codex_base_url,'key':self.cfg.codex_api_key,'model':self.cfg.codex_model,'prompt':prompt,'frames':frames,'schema':SCHEMA}),start,end)
                    budget.finish(request_id,'ok')
                except Exception:
                    budget.finish(request_id,'failed');raise
                segments.extend(result['segments']);unknown.extend(result['unknown']);reasons.append(result['reason'])
                self._update(ident,progress=window+1)
        segments=normalize_intervals(segments,duration) if segments else []
        unknown=normalize_intervals(unknown,duration) if unknown else []
        # 保守处理：不确定区域不得混入建议的喝水区间。
        for a,b in unknown:
            pieces=[]
            for start,end in segments:
                if end<=a or start>=b:pieces.append([start,end])
                else:
                    if start<a:pieces.append([start,a])
                    if end>b:pieces.append([b,end])
            segments=pieces
        result={'segments':segments,'unknown':unknown,'reason':'；'.join(reasons)[:3000],
                'model':self.cfg.codex_model,'sampling_seconds':.5,'duration':duration,'source':'codex_proposal'}
        self._update(ident,status='ready',result=json.dumps(result,ensure_ascii=False))
