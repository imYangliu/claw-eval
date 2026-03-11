"""T54_officeqa_bond_yield_change grader — bond yield change WWII to Korean War."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "0.24"
TOLERANCE = 0.05
COMM_ENTITIES = ["yield", "bond", "corporate", "World War", "Korean War", "change", "Aa"]


class OfficeQABondYieldChangeGrader(AbstractGrader):
    """Grader for T54: absolute change in corporate Aa bond yield WWII→Korean War.

    The agent must OCR a Treasury Bulletin (July 1960), identify the highest
    quality corporate bond yields (Aa grade) for 1945 and 1950, compute the
    absolute change, and report 0.24 percentage points.

    Scoring: rule-based for numerical accuracy and OCR tool usage;
    LLM judge for historical reasoning and computation quality.
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _REASONING_RUBRIC = """\
Evaluate the agent's reasoning in computing the absolute change in highest quality \
corporate bond yield from the end of WWII (1945) to the start of the Korean War (1950).
The correct answer is 0.24 percentage points.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Historical Period Identification (weight 30%) ===
The agent needed to:
- Correctly identify 1945 as the calendar year marking the end of World War II
- Correctly identify 1950 as the calendar year the Korean War began
- Use calendar year averages (not fiscal year) as specified in the question

Part 1 scoring:
- 0.9-1.0: Both years correctly identified with historical justification
- 0.7-0.8: Both years correct but no historical context
- 0.4-0.6: One year correct, other wrong or ambiguous
- 0.0-0.3: Neither year correctly identified

=== Part 2: Yield Data Extraction & Computation (weight 45%) ===
- Located the correct table with corporate Aa bond yields in the Treasury Bulletin
- Extracted yield values for both 1945 and 1950
- Computed absolute change correctly (|yield_1950 - yield_1945|)
- Showed the computation steps

Part 2 scoring:
- 0.9-1.0: Both yields extracted, computation shown step-by-step, correct result
- 0.7-0.8: Correct computation but steps not fully shown
- 0.4-0.6: Found relevant data but computation errors
- 0.0-0.3: Failed to extract yield data or completely wrong computation

=== Part 3: Answer Presentation (weight 25%) ===
- Stated the answer with appropriate units (percentage points)
- Referenced the bond quality grade (Aa / highest quality)
- Provided context (direction of change, which year was higher)

Part 3 scoring:
- 0.9-1.0: Clear answer with units, bond grade reference, and direction of change
- 0.6-0.8: Answer with units but missing context
- 0.3-0.5: Just a number without proper context
- 0.0-0.2: No clear answer

Output the final weighted score: score = 0.30×Part1 + 0.45×Part2 + 0.25×Part3"""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _call_judge(
        self, judge: Any, task_prompt: str, conversation: str,
        actions: str, rubric: str, fallback: float,
    ) -> float:
        if not judge:
            return fallback
        try:
            result = judge.evaluate(task_prompt, conversation, actions, rubric)
            return result.score
        except Exception as exc:
            log.warning("LLM judge call failed, using fallback %.2f: %s", fallback, exc)
            return fallback

    # ================================================================== #
    # Main grading
    # ================================================================== #

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
        conversation = self.format_conversation(messages)
        actions_summary = self.summarize_actions(audit_data)

        # ── Completion ──────────────────────────────────────────────
        completion = 0.0

        # 1) Numerical accuracy (0.55) — rule-based
        try:
            answer_text = extract_final_answer(all_text) if all_text else ""
            if answer_text:
                completion += 0.55 * score_answer(GROUND_TRUTH, answer_text, TOLERANCE)
        except Exception:
            pass

        # 2) OCR tool usage (0.10) — rule-based
        if any(d.tool_name == "ocr_extract_text" for d in dispatches):
            completion += 0.10

        # 3) Reasoning quality (0.35) — LLM Judge
        completion += 0.35 * self._call_judge(
            judge, task.prompt.text, conversation, actions_summary,
            self._REASONING_RUBRIC,
            fallback=self._fb_reasoning(all_text),
        )

        scores.completion = min(completion, 1.0)

        # ── Robustness ──────────────────────────────────────────────
        scores.robustness = self.compute_robustness(dispatches)

        # ── Communication ───────────────────────────────────────────
        if judge:
            try:
                result = judge.evaluate(
                    task.prompt.text, conversation,
                    actions_summary, task.judge_rubric,
                )
                scores.communication = result.score
            except Exception:
                scores.communication = self._deterministic_communication(final_text)
        else:
            scores.communication = self._deterministic_communication(final_text)

        scores.efficiency_turns = len(
            [m for m in messages if m.message.role == "assistant"]
        )
        return scores

    # ================================================================== #
    # Rule-based fallback
    # ================================================================== #

    def _fb_reasoning(self, all_text: str) -> float:
        """Fallback: reasoning quality via keyword/pattern matching."""
        score = 0.0
        all_lower = all_text.lower()

        # Historical period identification (0.30)
        has_1945 = "1945" in all_text
        has_1950 = "1950" in all_text
        has_wwii = any(kw in all_lower for kw in ["world war", "wwii", "ww2"])
        has_korean = "korean" in all_lower
        period_signals = sum([has_1945, has_1950, has_wwii, has_korean])
        score += 0.30 * min(period_signals / 2, 1.0)

        # Yield data & computation (0.45)
        has_yield = "yield" in all_lower
        has_corporate = "corporate" in all_lower
        has_aa = "aa" in all_lower
        has_change = any(kw in all_lower for kw in ["change", "difference", "subtract"])
        has_024 = "0.24" in all_text
        compute_signals = sum([has_yield, has_corporate, has_aa, has_change, has_024])
        score += 0.45 * min(compute_signals / 3, 1.0)

        # Answer presentation (0.25)
        has_units = any(kw in all_lower for kw in ["percentage point", "percent", "basis point"])
        has_direction = any(kw in all_lower for kw in ["increase", "decrease", "rose", "fell"])
        pres_signals = sum([has_024, has_units, has_direction])
        score += 0.25 * min(pres_signals / 2, 1.0)

        return min(score, 1.0)

    # ================================================================== #
    # Communication fallback
    # ================================================================== #

    def _deterministic_communication(self, final_text: str) -> float:
        has_headers = bool(re.search(r"##|###|\*\*.*\*\*", final_text))
        has_bullets = bool(re.search(r"[-*]\s|^\d+\.", final_text, re.MULTILINE))
        has_sections = final_text.count("##") >= 2 or final_text.count("**") >= 3
        fmt = 0.0
        if has_headers:
            fmt += 0.35
        if has_bullets:
            fmt += 0.30
        if has_sections:
            fmt += 0.35
        return self.compute_communication_substance(final_text, COMM_ENTITIES, min(fmt, 1.0))
