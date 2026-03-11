"""T45_video_mme_coauthor_papers grader — judge + anchor scoring."""

from __future__ import annotations

import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage


class VideoMmeCoauthorPapersGrader(AbstractGrader):
    """Grade Video-MME co-authored papers research."""

    ANCHOR_GROUPS = [
        [r"lei\s*li"],                                  # 4th author
        [r"shuhuai\s*ren"],                             # 5th author
        [r"temple"],                                    # AAAI paper
        [r"groundingme"],                               # CVPR paper
        [r"timechat"],                                  # ACM MM paper
        [r"video.?mme"],                                # CVPR paper (the paper itself)
        [r"vitatecs"],                                  # ECCV paper
        [r"fetv"],                                      # NeurIPS paper
        [r"openness.*clip|clip.*openness|delving"],     # ACL paper
        [r"cascadebert"],                               # EMNLP paper
        [r"dynamic.*knowledge.*distill"],               # EMNLP paper
        [r"text\s*auto\s*augment"],                     # EMNLP paper
    ]

    COMM_ENTITIES = [
        "Lei Li", "Shuhuai Ren",
        "TEMPLE", "GroundingME", "TimeChat", "Video-MME",
        "Vitatecs", "FETV", "CascadeBERT",
    ]

    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict] | None = None,
        judge: Any | None = None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict | None = None,
    ) -> DimensionScores:
        scores = DimensionScores()
        scores.safety = 1.0

        final_text = self._get_final_assistant_text(messages)
        all_text = self._get_all_assistant_text(messages)
        anchor_score = self._anchor_coverage_score(all_text)

        search_calls = [d for d in dispatches if d.tool_name == "web_search"]
        unique_searches = len({d.request_body.get("query", "") for d in search_calls})
        fetch_calls_count = len([d for d in dispatches if d.tool_name == "web_fetch"])
        search_effort = min((unique_searches + fetch_calls_count) / 8, 1.0)

        if judge:
            judged = judge.evaluate(
                task.prompt.text,
                self.format_conversation(messages),
                self.summarize_actions(audit_data),
                task.judge_rubric,
            ).score
            scores.completion = round(min(1.0, 0.70 * judged + 0.30 * search_effort), 2)
        else:
            scores.completion = round(min(1.0, 0.60 * anchor_score + 0.40 * search_effort), 2)

        # scores.communication = ...

        scores.robustness = self.compute_robustness(dispatches)
        scores.efficiency_turns = len([m for m in messages if m.message.role == "assistant"])
        return scores

    def _anchor_coverage_score(self, text: str) -> float:
        text_l = text.lower()
        covered = 0
        for group in self.ANCHOR_GROUPS:
            if any(re.search(pattern, text_l) for pattern in group):
                covered += 1
        return covered / len(self.ANCHOR_GROUPS)

    def _deterministic_communication(self, final_text: str, anchor_score: float) -> float:
        has_structure = bool(re.search(r"##|###|[-*]\s|^\d+\.", final_text, re.MULTILINE))
        format_score = 0.8 if has_structure else 0.6
        base = self.compute_communication_substance(final_text, self.COMM_ENTITIES, format_score)
        return round(min(1.0, 0.7 * base + 0.3 * anchor_score), 2)
