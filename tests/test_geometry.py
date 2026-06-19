from catcam.geometry import (
    area,
    intersection_area,
    ratio_rect_to_pixels,
    cat_overlaps_bowl,
)


def test_area():
    assert area((0, 0, 10, 4)) == 40


def test_intersection_partial():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert intersection_area(a, b) == 25


def test_intersection_none():
    a = (0, 0, 5, 5)
    b = (10, 10, 20, 20)
    assert intersection_area(a, b) == 0


def test_ratio_rect_to_pixels():
    assert ratio_rect_to_pixels((0.5, 0.5, 1.0, 1.0), 640, 480) == (320.0, 240.0, 640.0, 480.0)


def test_cat_overlaps_bowl_true_when_enough():
    bowl = (0, 0, 10, 10)          # 面积 100
    cat = (0, 0, 5, 10)            # 交集 50 -> 比例 0.5
    assert cat_overlaps_bowl(cat, bowl, min_ratio=0.15) is True


def test_cat_overlaps_bowl_false_when_too_little():
    bowl = (0, 0, 10, 10)          # 面积 100
    cat = (0, 0, 1, 10)            # 交集 10 -> 比例 0.1
    assert cat_overlaps_bowl(cat, bowl, min_ratio=0.15) is False
