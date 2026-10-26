"""复核队列口径：高把握的已确认结果可抽查，不强制人工重复确认。"""
from catcam.judge import Verdict, is_extreme_verdict


def needs_review(meta, candidate_only=False):
    if meta and meta.get('source') == 'human':
        return False
    if candidate_only or not meta or meta.get('source') not in ('local', 'ai'):
        return True
    score = meta.get('confidence')
    drinking = meta.get('is_drinking')
    if score is None or drinking is None:
        return True
    # 本地 confidence 是喝水得分；外部 AI confidence 是该类别的把握。
    if meta['source'] == 'ai':
        return not is_extreme_verdict(Verdict(True, score, '', 'ai'))
    return not is_extreme_verdict(Verdict(bool(drinking), score, '', meta['source']))


def count_pending_reviews(connection, start, end):
    # StatsStore 的独立用法也允许尚未经 FeedbackStore 扩展的旧 labels 表。
    columns = {row[1] for row in connection.execute('PRAGMA table_info(labels)')}
    source = "COALESCE(l.source,'human')" if 'source' in columns else "'human'"
    confidence = 'l.confidence' if 'confidence' in columns else 'NULL'
    rows = connection.execute(
        'SELECT e.clip_name,MAX(e.candidate_only),l.clip_name,l.is_drinking,'
        f'{source},{confidence} FROM events e LEFT JOIN labels l USING(clip_name) '
        'WHERE e.ts>=? AND e.ts<? AND e.review_candidate=1 GROUP BY e.clip_name',
        (start, end)).fetchall()
    return sum(needs_review({'is_drinking': drinking, 'source': origin, 'confidence': score}
                            if label else None, visual)
               for _, visual, label, drinking, origin, score in rows)
