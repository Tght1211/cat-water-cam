"""外部 Agent/Skill 素材接口：无需 SDK 或项目内 API Key，只接收待人工复核的建议。"""
from __future__ import annotations
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path
import cv2
from catcam.database import connect
from catcam.drink_intervals import normalize_intervals

SKILL_ROOT=Path(__file__).with_name('assets')/'skills'/'cat-water-cam'


def skill_archive():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SKILL_ROOT.rglob('*')):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                archive.write(path,'cat-water-cam/'+path.relative_to(SKILL_ROOT).as_posix())
    return buffer.getvalue()


def subtract_ranges(segments,excluded):
    for a,b in excluded:
        pieces=[]
        for start,end in segments:
            if end<=a or start>=b:pieces.append([start,end])
            else:
                if start<a:pieces.append([start,a])
                if end>b:pieces.append([b,end])
        segments=pieces
    return segments


class MaterialWorkspace:
    def __init__(self,feedback,clips_dir):
        self.feedback,self.clips_dir=feedback,Path(clips_dir)
        with connect(feedback.db_path) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS material_proposals (id TEXT PRIMARY KEY,clip TEXT NOT NULL,created REAL NOT NULL,result TEXT NOT NULL)')

    def clip(self,name):
        if not isinstance(name,str) or any(v in name for v in ('/','\\','..')):
            raise ValueError('录像名称无效')
        path=self.clips_dir/name
        if self.clips_dir.is_symlink() or path.is_symlink() or not path.is_file():raise ValueError('录像不存在')
        return path

    def metadata(self,name):
        path=self.clip(name);stat=path.stat()
        label=self.feedback.label_meta(name)
        if not label or label['is_drinking'] is not True:
            raise ValueError('Skill 仅可处理已确认喝水的视频，当前视频未确认或已被驳回')
        cap=cv2.VideoCapture(str(path))
        try:
            fps=cap.get(cv2.CAP_PROP_FPS)
            duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps>0 else 0
        finally:cap.release()
        if not math.isfinite(duration) or duration<=0:raise ValueError('无法读取录像时长')
        return {'clip':name,'duration':duration,'fps':fps,'bytes':stat.st_size,
                'fingerprint':hashlib.sha256(repr((name,stat.st_size,stat.st_mtime_ns)).encode()).hexdigest(),
                'video_url':'/clips/'+name,'label':label}

    def confirmed(self,recorder,limit=12,cursor=None):
        stored=self.feedback.clip_metadata()
        names=[path.name for path in recorder.list_clips() if (stored['meta'].get(path.name) or {}).get('is_drinking') is True]
        names.sort(reverse=True)
        if cursor:names=[name for name in names if name<cursor]
        selected=names[:limit]
        return {'scope':'confirmed_drinking_only','clips':[{'clip':name,'video_seconds':stored['durations'].get(name),
                'drinking_seconds':stored['meta'][name]['drinking_seconds'],
                'intervals':stored['meta'][name].get('drinking_intervals'),
                'source':stored['meta'][name]['source']} for name in selected],
                'next_cursor':selected[-1] if len(names)>limit else None}

    def propose(self,body):
        meta=self.metadata(body.get('clip'))
        if body.get('fingerprint')!=meta['fingerprint']:raise ValueError('录像已变化，请重新准备素材后提交建议')
        duration=meta['duration']
        coverage=normalize_intervals(body.get('coverage'),duration)
        segments=body.get('segments');unknown=body.get('unknown',[])
        if not isinstance(segments,list) or not isinstance(unknown,list):raise ValueError('建议和不确定区间须为数组')
        segments=normalize_intervals(segments,duration) if segments else []
        unknown=normalize_intervals(unknown,duration) if unknown else []
        if subtract_ranges(segments,coverage) or subtract_ranges(unknown,coverage):
            raise ValueError('建议区间超出声明的已分析范围')
        # 未分析的其余范围视作未知，不能把局部分析冒充整片结果。
        uncovered=subtract_ranges([[0,duration]],coverage)
        unknown=normalize_intervals(unknown+uncovered,duration) if unknown or uncovered else []
        segments=subtract_ranges(segments,unknown)
        reason=body.get('reason','');model=body.get('model','外部 Agent')
        if not isinstance(reason,str) or len(reason)>3000 or not isinstance(model,str) or len(model)>256:
            raise ValueError('说明或模型名称过长或格式无效')
        result={'clip':meta['clip'],'fingerprint':meta['fingerprint'],'duration':duration,'coverage':coverage,
                'segments':segments,'unknown':unknown,'reason':reason,'model':model,'source':'external_agent_proposal'}
        ident='skill_'+hashlib.sha256(json.dumps(result,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:32]
        with connect(self.feedback.db_path) as connection:
            connection.execute('BEGIN IMMEDIATE')
            current=connection.execute('SELECT is_drinking FROM labels WHERE clip_name=?',(meta['clip'],)).fetchone()
            if not current or not current[0]:raise ValueError('视频已被驳回，拒绝提交喝水轨建议')
            connection.execute('INSERT OR IGNORE INTO material_proposals VALUES(?,?,?,?)',
                               (ident,meta['clip'],time.time(),json.dumps(result,ensure_ascii=False)))
        return {'id':ident,'status':'ready','clip':meta['clip'],'result':result,'note':'仅保存建议，未改人工标注或训练状态'}

    def jobs(self,ident=None):
        with connect(self.feedback.db_path) as connection:
            rows=connection.execute('SELECT id,clip,created,result FROM material_proposals '+
                ('WHERE id=?' if ident else 'ORDER BY created DESC LIMIT 50'),(ident,) if ident else ()).fetchall()
        return [{'id':r[0],'clip':r[1],'created':r[2],'status':'ready','progress':1,'total':1,'result':json.loads(r[3]),'error':None} for r in rows]
