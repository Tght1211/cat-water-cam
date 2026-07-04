# 饮水机：滤芯提醒 + 剩余水量反推 + 喝水量(ml)自校准 设计

日期：2026-07-04

给「总览」页加一张**饮水机**卡片：滤芯更换倒计时、上次蓄水时间、按当期喝水次数**反推剩余水量**并提醒加水，
并估算小猫喝水量(ml)。

## 决策

- **ml 自校准**（用户选）：不测体积，靠「一整个蓄水周期的总量 ÷ 期间喝水次数」反推每次约几 ml，越用越准。
- **面板放总览页卡片**（用户选）。
- 提醒先做**网页卡片高亮**（该加水/该换滤芯变色 + 文案）；邮件提醒留作后续。

## 数据模型（`catcam/dispenser.py` · `DispenserStore`）

同一个 `data/catcam.db`，单行状态表 `dispenser`（id=1）：
- `last_filter_change_ts`：上次换滤芯时刻（0=未记录）
- `filter_cycle_days`：滤芯周期天数（默认 `cfg.filter_cycle_days=30`，网页可改）
- `last_refill_ts` / `last_refill_ml`：上次蓄水时刻 / 加了多少 ml（0=未记录）
- `calib_sum_ml` / `calib_drinks`：历史「已完结周期」的累计总量 / 累计喝水次数，
  **池化估计** `ml_per_drink = calib_sum_ml / calib_drinks`（未校准前用 `cfg.dispenser_default_ml_per_drink=30`）

## 核心逻辑（纯函数，可单测）

- `estimate_remaining(refill_ml, drinks_since, ml_per_drink) = max(0, refill_ml - drinks_since*ml_per_drink)`
- `filter_days_left(last_change_ts, cycle_days, now) = cycle_days - (now-last_change_ts)/86400`
- 自校准（在 `refill` 里）：新一次蓄水时，若上一次蓄水存在且期间喝过水，
  把「上次加的量」「上次期间喝水次数」并入池：`calib_sum_ml += last_refill_ml; calib_drinks += prev_drinks`，
  然后 `last_refill_ts=now; last_refill_ml=ml`。这样每完成一个「加满→再加满」周期就自动校准一次。

「期间喝水次数」由 web 层用 `StatsStore.count_between(last_refill_ts, now)`（既有口径：只数 `is_drinking=1`）
算好传进来——`DispenserStore` 不依赖 stats，保持单一职责。

## API（web.py）

- `GET /api/dispenser`：返回状态 + 派生量：`remaining_ml/remaining_pct/last_refill_ts/ml_per_drink/`
  `filter_cycle_days/last_filter_change_ts/filter_days_left/need_water/need_filter/`
  `drinks_since_refill/today_ml`（今日次数×ml_per_drink）。`need_water = 已记录蓄水 且 remaining_pct < cfg.dispenser_low_water_pct(0.2)`；
  `need_filter = 已记录换滤芯 且 filter_days_left <= 0`。
- `POST /api/dispenser/refill {ml}`：先用 stats 算上一周期喝水次数 → `store.refill(ml, now, prev_drinks)`。
- `POST /api/dispenser/filter`：`store.mark_filter(now)`。
- `POST /api/dispenser/config {filter_cycle_days}`：改滤芯周期。

## UI（总览页卡片）

一张「饮水机」卡片，放在总览网格里：
- **剩余水量**进度条（remaining_ml / last_refill_ml，百分比）+ 文案「约剩 X ml / 满 Y ml」；
  `need_water` 时进度条转琥珀/红 + 「该加水了」。未记录蓄水 → 引导「点『加满水』开始记录」。
- **上次蓄水**：相对时间（如「3 小时前」）。
- **滤芯**：`filter_days_left` 倒计时天数；`need_filter` 高亮「该换滤芯了」。
- **今日约喝**：`today_ml` ml（= 今日次数 × ml_per_drink），旁注「每次约 ml_per_drink ml（自校准）」。
- 按钮：**加满水**（弹个输入 ml，默认上次值或 2000）、**换了滤芯**、**滤芯周期**（改天数）。

## 测试

- `dispenser.py`：`estimate_remaining`/`filter_days_left` 纯函数；`DispenserStore` 用 tmp db 断言
  refill 自校准（跑两个周期后 `ml_per_drink` = 池化值）、mark_filter、set_cycle、默认值。
- web：`/api/dispenser` GET 初始态；refill 后 remaining 随喝水次数下降；filter 倒计时；need_water/need_filter 阈值。

`pytest -q` 全绿。
