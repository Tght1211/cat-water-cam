import math
import pytest
from catcam.drink_intervals import normalize_intervals
from catcam.feedback import FeedbackStore


def test_overlaps_merged_but_pauses_preserved():
    assert normalize_intervals([[8,9],[1,3],[2,4],[4,5]],10)==[[1,5],[8,9]]


@pytest.mark.parametrize('segments',[[],[[2,1]],[[0,11]],[[0,math.nan]],[[True,2]],[[0,math.inf]],[[-1,2]]])
def test_invalid_intervals_rejected(segments):
    with pytest.raises(ValueError):normalize_intervals(segments,10)


def test_store_persists_union_and_rejects_negative_labels(tmp_path):
    store=FeedbackStore(tmp_path/'test.db',tmp_path/'training')
    with store._conn() as connection:
        connection.execute("INSERT INTO labels(clip_name,is_drinking,source) VALUES('clip_1.mp4',1,'ai')")
    intervals,seconds=store.set_intervals('clip_1.mp4',[[1,3],[2,5],[7,8]],10)
    assert seconds==5
    meta=store.clip_metadata()['meta']['clip_1.mp4']
    assert meta['drinking_intervals']==[[1,5],[7,8]]
    assert meta['drinking_seconds']==5 and meta['source']=='human'
    store.set_duration('clip_1.mp4',3)
    assert 'drinking_intervals' not in store.clip_metadata()['meta']['clip_1.mp4']
    with store._conn() as connection:
        connection.execute("UPDATE labels SET is_drinking=0 WHERE clip_name='clip_1.mp4'")
    with pytest.raises(ValueError):store.set_intervals('clip_1.mp4',[[1,2]],10)
