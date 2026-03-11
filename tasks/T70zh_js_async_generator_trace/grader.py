"""T44_js_async_generator_trace grader — judge + anchor scoring."""

from __future__ import annotations

import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage


class JsAsyncGeneratorTraceGrader(AbstractGrader):
    """Grade JS async generator + Proxy + microtask execution order analysis."""

    ANCHOR_GROUPS = [
        [r"thenable"],                                      # scheduler is Thenable
        [r"global:init"],                                   # sync start
        [r"p:a:start"],                                     # procA starts
        [r"sched:then"],                                    # scheduler.then called
        [r"p:b:start"],                                     # procB starts
        [r"global:end"],                                    # sync end
        [r"inter:1"],                                       # Promise chain step 1
        [r"sched:internal"],                                # scheduler internal microtask
        [r"sched:resolve"],                                 # scheduler resolve
        [r"microtask|微任务"],                               # microtask concept
        [r"generator|生成器"],                               # generator concept
        [r"p:a:end.*不会|不会.*p:a:end|won'?t.*print"],     # P:A:End won't print
    ]

    COMM_ENTITIES = [
        "Thenable", "microtask", "generator",
        "Global:Init", "Sched:Then", "Global:End",
        "Inter:1", "Sched:Resolve",
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

        # Sandbox effort (no web search for this task)
        sandbox_calls = len([d for d in dispatches if d.tool_name == "sandbox_shell_exec"])
        tool_effort = min(sandbox_calls / 1, 1.0)

        if judge:
            judged = judge.evaluate(
                task.prompt.text,
                self.format_conversation(messages),
                self.summarize_actions(audit_data),
                task.judge_rubric,
            ).score
            scores.completion = round(min(1.0, 0.70 * judged + 0.30 * tool_effort), 2)
        else:
            scores.completion = round(min(1.0, 0.70 * anchor_score + 0.30 * tool_effort), 2)

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
