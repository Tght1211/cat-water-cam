#!/usr/bin/env python3
"""仅处理已确认喝水视频：列素材、准备连续抽帧、提交喝水轨草稿。标准库 + ffmpeg。"""
import argparse
import hashlib
import http.cookiejar
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode,urlsplit,quote
from urllib.request import Request,build_opener,ProxyHandler,HTTPCookieProcessor,HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class Client:
    def __init__(self,base):
        parsed=urlsplit(base)
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('服务地址格式无效')
        if parsed.hostname!='localhost':
            try:address=ipaddress.ip_address(parsed.hostname)
            except ValueError:raise ValueError('请使用本机 localhost 或明确的内网 IP 地址')
            if not (address.is_loopback or address.is_private):raise ValueError('素材接口仅连接本机或内网地址')
        self.base=base.rstrip('/')
        self.opener=build_opener(ProxyHandler({}),NoRedirect(),HTTPCookieProcessor(http.cookiejar.CookieJar()))
        password=os.environ.get('CATCAM_PASSWORD')
        if password:self.json('/api/auth/login',{'password':password})

    def json(self,path,body=None):
        request=Request(self.base+path,data=json.dumps(body).encode() if body is not None else None,
                        headers={'Content-Type':'application/json'} if body is not None else {})
        try:
            with self.opener.open(request,timeout=30) as response:
                raw=response.read(8*1024*1024+1)
        except HTTPError as error:
            if error.code in (401,403):raise ValueError('项目需要登录；在环境变量 CATCAM_PASSWORD 中提供访问密码')
            try:message=json.loads(error.read(4096)).get('detail','接口请求失败')
            except ValueError:message='接口请求失败'
            raise ValueError(str(message))
        if len(raw)>8*1024*1024:raise ValueError('接口响应过大')
        return json.loads(raw)

    def download(self,path,target,max_bytes):
        temp=target.with_suffix('.part')
        try:
            with self.opener.open(self.base+path,timeout=60) as response,open(temp,'wb') as output:
                os.chmod(temp,0o600)
                size=0
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>max_bytes:raise ValueError('下载超出视频大小限制')
                    output.write(chunk)
            temp.replace(target)
        finally:
            if temp.exists():temp.unlink()


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    path.chmod(0o600)


def prepare(client,args):
    meta=client.json('/api/materials/'+quote(args.clip,safe=''))
    start=args.start;end=min(meta['duration'],start+args.seconds)
    if not math.isfinite(start) or not math.isfinite(args.seconds) or not math.isfinite(args.step) or not 0<=start<end or not .25<=args.step<=2 or not 0<args.seconds<=60:
        raise ValueError('起点须在录像内；窗口 0–60 秒，采样间隔 0.25–2 秒')
    root=Path(args.out).expanduser().resolve();root.mkdir(mode=0o700,parents=True,exist_ok=True)
    source=root/'source.json';video=root/'source.mp4'
    if source.exists():
        saved=json.loads(source.read_text())
        if saved.get('fingerprint')!=meta['fingerprint']:raise ValueError('目录中是其他版本素材，请更换输出目录')
    else:save_json(source,meta)
    if not video.exists():
        client.download(meta['video_url'],video,max_bytes=2*1024*1024*1024)
        current=client.json('/api/materials/'+quote(args.clip,safe=''))
        if current['fingerprint']!=meta['fingerprint']:
            video.unlink();raise ValueError('下载期间录像变化，请重新准备')
    digest=hashlib.sha256(f'{start}:{end}:{args.step}'.encode()).hexdigest()[:12]
    directory=root/('window_'+digest);directory.mkdir(mode=0o700,exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists():return {'manifest':str(manifest),'cached':True}
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise ValueError('准备画面需要 ffmpeg；请先安装或在项目运行环境中执行')
    count=max(1,math.ceil((end-start)/args.step))
    command=[ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(video),
             '-t',str(end-start),'-vf',f'fps={1/args.step},scale=640:-2','-frames:v',str(count),
             '-q:v','3',str(directory/'frame_%04d.jpg')]
    subprocess.run(command,check=True,capture_output=True,timeout=180)
    images=sorted(directory.glob('frame_*.jpg'))
    if not images:raise ValueError('没有抽取到画面')
    for image in images:image.chmod(0o600)
    data={'clip':args.clip,'fingerprint':meta['fingerprint'],'duration':meta['duration'],'coverage':[[start,end]],
          'sampling_seconds':args.step,'frames':[{'time':min(end,start+i*args.step),'path':str(image)} for i,image in enumerate(images)],
          'segments':[],'unknown':[],'reason':'','model':'外部 Agent'}
    save_json(manifest,data)
    return {'manifest':str(manifest),'frames':len(images),'coverage':data['coverage'],
            'note':'请查看连续帧后填写独立 proposal.json；脚本不会自行识别或改变已保存时长'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default=os.environ.get('CATCAM_URL','http://127.0.0.1:8000'))
    commands=parser.add_subparsers(dest='command',required=True)
    ls=commands.add_parser('list',help='只列已确认喝水视频');ls.add_argument('--limit',type=int,default=12);ls.add_argument('--cursor')
    meta=commands.add_parser('info');meta.add_argument('clip')
    prep=commands.add_parser('prepare');prep.add_argument('clip');prep.add_argument('--out',required=True)
    prep.add_argument('--start',type=float,default=0);prep.add_argument('--seconds',type=float,default=30);prep.add_argument('--step',type=float,default=.5)
    submit=commands.add_parser('propose',help='只保存草稿，不覆盖人工区间');submit.add_argument('file')
    commands.add_parser('proposals');commands.add_parser('calibration')
    args=parser.parse_args(argv);client=Client(args.base)
    if args.command=='list':
        params={'limit':args.limit}
        if args.cursor:params['cursor']=args.cursor
        result=client.json('/api/materials?'+urlencode(params))
    elif args.command=='info':result=client.json('/api/materials/'+quote(args.clip,safe=''))
    elif args.command=='prepare':result=prepare(client,args)
    elif args.command=='propose':
        path=Path(args.file)
        if path.stat().st_size>128*1024:raise ValueError('建议文件过大')
        body=json.loads(path.read_text())
        fields=('clip','fingerprint','coverage','segments','unknown','reason','model')
        result=client.json('/api/materials/proposals',{key:body[key] for key in fields if key in body})
    elif args.command=='proposals':result=client.json('/api/materials/proposals')
    else:result=client.json('/api/drinking-calibration')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print(json.dumps({'error':str(error)},ensure_ascii=False),file=sys.stderr)
        sys.exit(1)
