"""局域网访问：密码摘要、签名会话及所有资源的登录保护。"""
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE='catcam_session'
SESSION_SECONDS=12*3600


def hash_password(password):
    salt=secrets.token_hex(16)
    digest=hashlib.scrypt(password.encode(),salt=bytes.fromhex(salt),n=16384,r=8,p=1).hex()
    return f'scrypt${salt}${digest}'


def verify_password(password,encoded):
    if not isinstance(password,str) or not 1<=len(password)<=256:
        return False
    try:
        algorithm,salt,digest=encoded.split('$')
        if algorithm!='scrypt':return False
        actual=hashlib.scrypt(password.encode(),salt=bytes.fromhex(salt),n=16384,r=8,p=1).hex()
        return hmac.compare_digest(actual,digest)
    except (ValueError,TypeError):
        return False


def install_access(app,password_hash,session_secret,password_update=None):
    if not password_hash:
        return
    if not session_secret:
        raise ValueError('启用访问密码时需要会话密钥')
    signing_key=hashlib.sha256((session_secret+password_hash).encode()).digest()
    revoked={}
    attempts=defaultdict(deque)
    lock=threading.Lock()

    def sign(payload):
        return hmac.new(signing_key,payload.encode(),hashlib.sha256).hexdigest()

    def valid(token):
        if not token or len(token)>256:return False
        try:
            timestamp,nonce,signature=token.split('.')
            age=time.time()-int(timestamp)
            if not 0<=age<=SESSION_SECONDS or len(nonce)!=32:return False
            if not hmac.compare_digest(sign(timestamp+'.'+nonce),signature):return False
            with lock:return token not in revoked
        except (ValueError,TypeError):return False

    app.state.access_session_valid=valid

    @app.middleware('http')
    async def require_login(request:Request,call_next):
        public=(request.url.path=='/login' and request.method=='GET') or (request.url.path=='/api/auth/login' and request.method=='POST')
        if request.method not in ('GET','HEAD','OPTIONS'):
            origin=request.headers.get('origin')
            if origin and origin!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'不允许跨站操作'},status_code=403)
        if not public and not valid(request.cookies.get(COOKIE)):
            if request.url.path=='/':return RedirectResponse('/login',status_code=303,headers={'Cache-Control':'no-store'})
            return JSONResponse({'detail':'请先输入访问密码'},status_code=401,headers={'Cache-Control':'no-store'})
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['X-Frame-Options']='DENY'
        return response

    @app.get('/login')
    def login_page(request:Request):
        if valid(request.cookies.get(COOKIE)):return RedirectResponse('/',status_code=303)
        return HTMLResponse(Path(__file__).with_name('static').joinpath('login.html').read_text())

    @app.post('/api/auth/login')
    async def login(request:Request):
        ip=request.client.host if request.client else 'unknown'
        now=time.time()
        with lock:
            for address in list(attempts):
                while attempts[address] and now-attempts[address][0]>300:attempts[address].popleft()
                if not attempts[address]:del attempts[address]
            if len(attempts[ip])>=10 or sum(map(len,attempts.values()))>=100:
                return JSONResponse({'detail':'尝试次数过多，请稍后重试'},status_code=429)
            attempts[ip].append(now)
        try:body=await request.json()
        except ValueError:return JSONResponse({'detail':'请求格式无效'},status_code=400)
        if not isinstance(body,dict) or not verify_password(body.get('password'),password_hash):
            return JSONResponse({'detail':'访问密码不正确'},status_code=401)
        payload=f'{int(now)}.{secrets.token_hex(16)}'
        response=JSONResponse({'ok':True})
        response.set_cookie(COOKIE,payload+'.'+sign(payload),max_age=SESSION_SECONDS,httponly=True,
                            secure=request.url.scheme=='https',samesite='strict',path='/')
        return response

    @app.post('/api/auth/password')
    async def change_password(request:Request):
        nonlocal password_hash,signing_key
        if password_update is None:
            return JSONResponse({'detail':'当前进程不支持保存访问密码'},status_code=400)
        try:body=await request.json()
        except ValueError:return JSONResponse({'detail':'请求格式无效'},status_code=400)
        if not isinstance(body,dict):
            return JSONResponse({'detail':'请求格式无效'},status_code=400)
        new=body.get('new_password')
        if not isinstance(new,str) or not 8<=len(new)<=256:
            return JSONResponse({'detail':'新密码需要8到256个字符'},status_code=400)
        if new!=body.get('confirm_password'):
            return JSONResponse({'detail':'两次输入的新密码不一致'},status_code=400)
        with lock:
            ip=request.client.host if request.client else 'unknown'
            now=time.time()
            while attempts[ip] and now-attempts[ip][0]>300:attempts[ip].popleft()
            if len(attempts[ip])>=10:
                return JSONResponse({'detail':'尝试次数过多，请稍后重试'},status_code=429)
            if not valid_for_change(request.cookies.get(COOKIE)):
                return JSONResponse({'detail':'请重新登录后再修改密码'},status_code=401)
            if not verify_password(body.get('current_password'),password_hash):
                attempts[ip].append(now)
                return JSONResponse({'detail':'当前密码不正确'},status_code=400)
            encoded=hash_password(new)
            try:password_update(encoded)
            except Exception:
                return JSONResponse({'detail':'密码保存失败，请重试'},status_code=500)
            password_hash=encoded
            signing_key=hashlib.sha256((session_secret+encoded).encode()).digest()
            revoked.clear()
            attempts.pop(ip,None)
            payload=f'{int(now)}.{secrets.token_hex(16)}'
            response=JSONResponse({'ok':True})
            response.set_cookie(COOKIE,payload+'.'+sign(payload),max_age=SESSION_SECONDS,httponly=True,
                                secure=request.url.scheme=='https',samesite='strict',path='/')
            return response

    def valid_for_change(token):
        # 调用者持有锁，避免valid()再次获取同一个锁。
        if not token or len(token)>256:return False
        try:
            timestamp,nonce,signature=token.split('.')
            return (0<=time.time()-int(timestamp)<=SESSION_SECONDS and len(nonce)==32
                    and hmac.compare_digest(sign(timestamp+'.'+nonce),signature) and token not in revoked)
        except (ValueError,TypeError):return False

    @app.post('/api/auth/logout')
    def logout(request:Request):
        token=request.cookies.get(COOKIE)
        with lock:
            now=time.time()
            for old,expiry in list(revoked.items()):
                if expiry<now:del revoked[old]
            if token:revoked[token]=now+SESSION_SECONDS
        response=JSONResponse({'ok':True});response.delete_cookie(COOKIE,path='/')
        return response
