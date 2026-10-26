import pytest

from catcam.review_policy import needs_review


@pytest.mark.parametrize('source,drinking,score,pending', [
    ('local', False, 0.0, False),
    ('local', False, .005, False),
    ('local', False, .006, True),
    ('local', False, .99, True),  # 得分与类别冲突不能自动放过
    ('local', True, .995, False),
    ('local', True, .8, True),
    ('ai', False, .999, False),  # AI 分数是类别把握，不是喝水概率
    ('ai', False, .995, False),
    ('ai', False, .1, True),
    ('local', False, None, True),
    ('local', False, float('nan'), True),
    ('local', False, -1, True),
])
def test_only_valid_high_confidence_labels_skip_mandatory_review(source, drinking, score, pending):
    assert needs_review({'source': source, 'is_drinking': drinking, 'confidence': score}) == pending


def test_missing_label_or_bad_visibility_requires_review():
    assert needs_review(None)
    assert needs_review({'source':'local', 'is_drinking':False, 'confidence':0}, candidate_only=True)
    assert not needs_review({'source':'human', 'is_drinking':False}, candidate_only=True)
