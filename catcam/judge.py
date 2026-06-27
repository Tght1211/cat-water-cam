"""整段视频裁判：录完一段后判「猫是否真喝水」，作为「发邮件 + 记次数」的唯一权威。

第一阶段实现 VLMClipJudge——包住 ai_labeler（外部视觉大模型，已具模型池轮换/兜底）。
第二阶段会再加 LocalVideoClipJudge（本地 VideoMAE 冻结特征 + 小头），同一 judge() 接口。
判定 fail-open：拿不到判定 → 该段不发邮件、不计数、保持未标注（绝不污染训练集 / 误发）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Verdict:
    drinking: bool
    confidence: float | None
    reason: str
    by: str            # 哪个裁判判的：'vlm' / 本地模型版本号等


class VLMClipJudge:
    """用外部视觉大模型当裁判：调 ai_labeler 判一段并写标注，适配成 Verdict。

    ai_labeler.label(clip) 返回 {"drinking","confidence","reason"} 或 None（人工已标/抽帧空），
    全模型失败时 raise——这里吞成 None（fail-open）。标注写入由 ai_labeler 内部完成。
    """

    def __init__(self, ai_labeler, log=print):
        self.ai_labeler = ai_labeler
        self.log = log

    def judge(self, clip_path) -> Verdict | None:
        try:
            result = self.ai_labeler.label(clip_path)
        except Exception as e:  # noqa: BLE001 —— 裁判失败绝不能崩采集/误发
            self.log(f"AI 裁判失败（{clip_path}）：{e}")
            return None
        if result is None:
            return None
        return Verdict(
            drinking=bool(result["drinking"]),
            confidence=result.get("confidence"),
            reason=str(result.get("reason", "") or ""),
            by="vlm",
        )


def judge_and_notify(judge, emailer, stats, clip_path, start_ts: float, photo, log=print):
    """对一段跑裁判；判「真喝水」且有照片才发邮件。返回 Verdict 或 None。

    纯流程、无线程——调用方负责放后台线程。写标注在 judge.judge() 内部已完成。
    """
    verdict = judge.judge(clip_path)
    if verdict is not None and verdict.drinking and photo is not None:
        emailer.notify_drinking(stats, photo, datetime.fromtimestamp(start_ts))
    return verdict


def route_clip(*, clip_path, start_ts: float, photo, ai_labeler, local_judge, mode: str,
               emailer, stats, feedback, log=print) -> dict:
    """会话录完后按模式编排：谁标注 / 谁判邮件计数 / 谁影子预测。返回各动作结果（供测试/日志）。

    - gate + 有本地模型：本地当权威，写 source='local' 计数标签，**不调 VLM**（画面不出本机）。
    - 否则（shadow / 无本地）：VLM 当权威+标注（第一阶段行为）；有本地模型则只影子预测、回填 events。
    """
    name = Path(clip_path).name
    authority = None
    emailed = False
    shadow_pred = None

    if mode == "gate" and local_judge is not None:
        authority = local_judge.judge(clip_path)
        if authority is not None:
            feedback.record_machine_label(
                name, authority.drinking, source="local",
                confidence=authority.confidence, reason=authority.reason,
            )
    else:
        if ai_labeler is not None:
            authority = VLMClipJudge(ai_labeler, log).judge(clip_path)
        if local_judge is not None:
            v = local_judge.judge(clip_path)
            if v is not None:
                stats.set_prediction(name, int(v.drinking), local_judge.version)
                shadow_pred = v.drinking

    if authority is not None and authority.drinking and photo is not None:
        emailer.notify_drinking(stats, photo, datetime.fromtimestamp(start_ts))
        emailed = True
    return {"authority": authority, "emailed": emailed, "shadow_pred": shadow_pred}
