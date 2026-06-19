from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrinkingEvent:
    timestamp: float


class DrinkingDetector:
    def __init__(self, dwell_seconds: float, cooldown_seconds: float):
        self.dwell_seconds = dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self._dwell_start: float | None = None
        self._cooldown_until: float = 0.0

    def update(self, now: float, cat_in_roi: bool) -> DrinkingEvent | None:
        if not cat_in_roi:
            self._dwell_start = None
            return None
        if now < self._cooldown_until:
            # 冷却期内：不计时、不触发
            self._dwell_start = None
            return None
        if self._dwell_start is None:
            self._dwell_start = now
            return None
        if now - self._dwell_start >= self.dwell_seconds:
            self._dwell_start = None
            self._cooldown_until = now + self.cooldown_seconds
            return DrinkingEvent(timestamp=now)
        return None
