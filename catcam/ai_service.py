"""OpenAI兼容视觉老师的配置校验及手动测试；测试不保存配置、不写训练标签。"""
from __future__ import annotations

import base64
import time
from urllib.parse import urlsplit

import cv2
import httpx

from catcam.ai_budget import AIBudget


def resolve_settings(body, cfg):
    base = body.get('ai_base_url', cfg.ai_base_url)
    if not isinstance(base, str):
        raise ValueError('Base URL 必须是 HTTP 或 HTTPS 地址')
    base = base.strip().rstrip('/')
    try:
        parsed = urlsplit(base)
        valid = parsed.scheme in ('http', 'https') and bool(parsed.hostname) and not (
            parsed.username or parsed.password or parsed.query or parsed.fragment)
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Base URL 必须是 HTTP 或 HTTPS 地址，且不能包含密钥、查询参数或片段')
    model = body.get('ai_model', cfg.ai_model)
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise ValueError('请填写有效的视觉模型名称')
    value = body.get('ai_daily_budget', cfg.ai_daily_budget)
    try:
        limit = int(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError('每日调用上限必须为正整数') from e
    if isinstance(value, bool) or limit < 1 or str(value).strip() != str(limit):
        raise ValueError('每日调用上限必须为正整数')
    key = body.get('ai_api_key') or cfg.ai_api_key
    if not isinstance(key, str):
        raise ValueError('API Key 必须为文本')
    return {'base_url': base, 'model': model.strip(), 'limit': limit, 'key': key.strip()}


def test_service(body, cfg, frame, db_path, client=None):
    settings = resolve_settings(body, cfg)
    if not settings['key']:
        raise ValueError('请先填写 API Key')
    kind = body.get('kind', 'connection')
    if kind not in ('connection', 'vision'):
        raise ValueError('请选择连接测试或视觉测试')
    request_id = None
    budget = None
    content = None
    if kind == 'vision':
        if frame is None:
            raise ValueError('摄像头尚无画面，暂不能测试视觉推理')
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            frame = cv2.resize(frame, (max(1, round(w * 640 / max(h, w))), max(1, round(h * 640 / max(h, w)))))
        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            raise ValueError('当前画面编码失败')
        content = [{'type': 'text', 'text': '请用一句简短中文描述这张摄像头图片中看到的主要物体。只描述可见内容。'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()}}]
        budget = AIBudget(db_path, settings['limit'], cfg.ai_min_interval_seconds)
        request_id = budget.reserve(settings['model'], '__manual_vision_test__')
        if request_id is None:
            return {'ok': False, 'kind': kind, 'message': '今日调用预算已用完或仍在调用冷却，请稍后再试。', 'budget': budget.status()}
    owned = client is None
    if owned:
        client = httpx.Client(trust_env=False, timeout=45)
    started = time.monotonic()
    try:
        headers = {'Authorization': 'Bearer ' + settings['key']}
        if kind == 'connection':
            response = client.get(settings['base_url'] + '/models', headers=headers)
        else:
            response = client.post(settings['base_url'] + '/chat/completions', headers=headers,
                                   json={'model': settings['model'], 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 256})
        if response.is_error:
            if budget:
                budget.finish(request_id, 'failed')
            messages = {401: 'API Key 未通过认证。', 403: '服务拒绝访问，请检查权限。',
                        404: '接口或模型不存在，请检查 Base URL 和模型名称。', 429: '上游调用限额或频率限制，请稍后重试。'}
            return {'ok': False, 'kind': kind, 'http_status': response.status_code,
                    'message': messages.get(response.status_code, '上游返回错误，暂未确认服务可用。')}
        data = response.json()
        if kind == 'connection':
            models = [m['id'] for m in data.get('data', []) if isinstance(m, dict) and isinstance(m.get('id'), str)]
            if not models:
                return {'ok': False, 'kind': kind, 'message': '接口已响应，但未返回有效模型列表。'}
            result = {'ok': True, 'kind': kind, 'model_count': len(models), 'models': models,
                      'model_available': settings['model'] in models,
                      'message': f'连接成功，获取到 {len(models)} 个模型；视觉推理需单独测试。'}
        else:
            choices = data.get('choices') or []
            text = choices[0].get('message', {}).get('content') if choices else None
            if not isinstance(text, str) or not text.strip():
                budget.finish(request_id, 'failed')
                return {'ok': False, 'kind': kind, 'message': '上游未返回有效视觉回复，暂未确认推理可用。'}
            budget.finish(request_id, 'success')
            result = {'ok': True, 'kind': kind, 'message': '视觉推理成功，已收到当前画面的回复。',
                      'reply': text.replace(settings['key'], '[已隐藏]').strip()[:500]}
        result['latency_ms'] = round((time.monotonic() - started) * 1000)
        if budget:
            result['budget'] = budget.status()
        return result
    except (httpx.HTTPError, ValueError, TypeError, AttributeError, IndexError) as error:
        if budget:
            budget.finish(request_id, 'failed')
        # 不返回异常原文，避免代理回显认证信息。
        message = ('上游请求超时，暂未确认服务可用。' if isinstance(error, httpx.TimeoutException)
                   else '网络连接失败，请检查服务地址。' if isinstance(error, httpx.HTTPError)
                   else '上游返回格式无效，暂未确认服务可用。')
        return {'ok': False, 'kind': kind, 'message': message}
    finally:
        if owned:
            client.close()
