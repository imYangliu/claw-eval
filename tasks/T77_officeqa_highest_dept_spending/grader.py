"""T51_officeqa_highest_dept_spending grader — highest spending dept FY1955."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "36080"
TOLERANCE = 0.05
DEPT_ANSWER = "Defense"
DOMAIN_KEYWORDS = ["department", "defense", "spending", "fiscal", "1955", "million"]
COMM_ENTITIES = ["Defense", "1955", "36,080", "million", "department", "spending"]


class OfficeQAHighestDeptSpendingGrader(AbstractGrader):
    """Grader for T51: highest spending U.S. Federal Department FY1955.

    The agent must OCR a Treasury Bulletin scan, find the department expenditure
    table for FY1955, identify that Defense is the highest spender, and report
    36,080 million dollars.

    Scoring: rule-based for numerical accuracy, department identification,
    and OCR tool usage; LLM judge for reasoning quality.
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _REASONING_QUALITY_RUBRIC = """\
Evaluate the quality of the agent's reasoning in finding the highest spending \
U.S. Federal Department in FY1955 from a Treasury Bulletin.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Department Identification & Comparison (weight 45%) ===
The agent needed to compare spending across multiple departments and identify \
the highest one (Department of Defense at 36,080 million):
- Did it explicitly state that Defense / Department of Defense had the highest spending?
- Did it list or compare multiple departments' spending figures to justify this?
- Did it show the comparison logic (not just assert the answer)?

Part 1 scoring:
- 0.9-1.0: Clearly identified Defense as highest, compared with other departments' \
figures to demonstrate this
- 0.7-0.8: Identified Defense as highest with some supporting comparison
- 0.5-0.6: Identified Defense as highest but no comparison shown
- 0.2-0.4: Mentioned Defense but didn't clearly state it was the highest
- 0.0-0.1: Didn't identify the department or named wrong one

=== Part 2: Data Source & Extraction (weight 25%) ===
- Referenced the Treasury Bulletin (October 1958) as data source
- Identified the correct table/section (department expenditures, FY1955)
- Distinguished fiscal year from calendar year

Part 2 scoring:
- 0.9-1.0: Clear data source reference and table identification
- 0.6-0.8: Mentioned source but table identification unclear
- 0.3-0.5: Gave answer without explaining source
- 0.0-0.2: No source reference

=== Part 3: Answer Presentation (weight 30%) ===
- Stated both the department name AND the amount with units
- Provided context (fiscal year 1955, millions of dollars)
- Clear and precise final answer

Part 3 scoring:
- 0.9-1.0: Complete answer (dept name + amount + units + context)
- 0.6-0.8: Has dept and amount but missing units or context
- 0.3-0.5: Only number or only department name
- 0.0-0.2: No clear answer

Output the final weighted score: score = 0.45×Part1 + 0.25×Part2 + 0.30×Part3"""

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

        # 3) Reasoning quality incl. dept identification (0.35) — LLM Judge
        completion += 0.35 * self._call_judge(
            judge, task.prompt.text, conversation, actions_summary,
            self._REASONING_QUALITY_RUBRIC,
            fallback=self._fb_reasoning_quality(all_text),
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

    def _fb_reasoning_quality(self, all_text: str) -> float:
        """Fallback: reasoning quality via keyword matching."""
        score = 0.0
        all_lower = all_text.lower()

        # Department identification & comparison (0.45)
        dept_names = ["defense", "treasury", "agriculture", "commerce", "interior",
                      "justice", "labor", "state", "hew", "health"]
        depts_mentioned = sum(1 for d in dept_names if d in all_lower)
        if depts_mentioned >= 3 and "defense" in all_lower:
            score += 0.45
        elif depts_mentioned >= 2 and "defense" in all_lower:
            score += 0.30
        elif "defense" in all_lower:
            score += 0.15
        elif depts_mentioned >= 1:
            score += 0.05

        # Data source (0.25)
        source_kw = ["treasury", "bulletin", "ocr", "table", "1958"]
        source_count = sum(1 for kw in source_kw if kw in all_lower)
        score += 0.25 * min(source_count / 2, 1.0)

        # Answer presentation (0.30)
        has_dept = "defense" in all_lower
        has_amount = "36" in all_text and ("080" in all_text or "080" in all_text.replace(",", ""))
        has_units = "million" in all_lower
        pres_count = sum([has_dept, has_amount, has_units])
        score += 0.30 * (pres_count / 3)

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
