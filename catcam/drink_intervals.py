"""人工喝水动作区间：合并交集，避免重叠时重复计时。"""
import math


def normalize_intervals(segments, duration):
    if not isinstance(segments, list) or not 0 < len(segments) <= 200:
        raise ValueError('请标注 1 至 200 段喝水区间；没有喝水请驳回记录')
    result = []
    for pair in segments:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('每段须包含开始和结束秒数')
        start, end = pair
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair):
            raise ValueError('区间时间须为有限数字')
        if not 0 <= start < end <= duration:
            raise ValueError('区间须位于录像内，且结束晚于开始')
        result.append([float(start), float(end)])
    merged = []
    for start, end in sorted(result):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
