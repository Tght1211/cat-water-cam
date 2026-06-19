from __future__ import annotations

Rect = tuple[float, float, float, float]


def area(rect: Rect) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def ratio_rect_to_pixels(rect: Rect, width: int, height: int) -> Rect:
    x1, y1, x2, y2 = rect
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def cat_overlaps_bowl(cat: Rect, bowl: Rect, min_ratio: float) -> bool:
    bowl_area = area(bowl)
    if bowl_area <= 0:
        return False
    return intersection_area(cat, bowl) / bowl_area >= min_ratio
