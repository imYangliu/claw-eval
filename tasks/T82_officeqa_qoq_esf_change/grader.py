"""T56_officeqa_qoq_esf_change grader — QoQ percent change in ESF total assets 2022."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "4.815"
TOLERANCE = 0.05
COMM_ENTITIES = ["QoQ", "percent", "ESF", "Exchange Stabilization", "assets", "2022", "quarter"]


class OfficeQAQoQESFChangeGrader(AbstractGrader):
    """Grader for T56: QoQ percent change in ESF total assets.

    The agent must OCR a Treasury Bulletin (December 2022), locate the Exchange
    Stabilization Fund balance sheet, extract total assets for end of Q2 (June)
    and Q3 (September) 2022, compute the absolute QoQ percent change, and
    report 4.815% rounded to the nearest thousandth.

    Scoring: rule-based for numerical accuracy and OCR tool usage;
    LLM judge for computation methodology quality.
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _METHODOLOGY_RUBRIC = """\
Evaluate the agent's methodology in computing the absolute QoQ percent change in \
total assets of the Exchange Stabilization Fund (ESF) from end of June 2022 to \
end of September 2022.
The correct answer is 4.815%.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: QoQ Computation Method (weight 40%) ===
The agent needed to:
- Apply the correct QoQ percent change formula: |(Q3 - Q2) / Q2| × 100
- Extract the correct Q2 (June 30) and Q3 (September 30) total asset values
- Show the computation with actual numbers
- Round to the nearest thousandth (3 decimal places)

Part 1 scoring:
- 0.9-1.0: Correct formula, both values shown, computation step-by-step, proper rounding
- 0.7-0.8: Correct formula and result but steps abbreviated
- 0.5-0.6: Right approach but minor computational or rounding errors
- 0.2-0.4: Attempted percent change but wrong formula or wrong values
- 0.0-0.1: No computation attempted

=== Part 2: ESF Data Identification (weight 35%) ===
- Correctly identified the Exchange Stabilization Fund section in the bulletin
- Found the balance sheet / total assets table
- Extracted values for the correct time periods (Q2 and Q3 2022)
- Understood "end of June" = Q2 end and "end of September" = Q3 end

Part 2 scoring:
- 0.9-1.0: Correct section, correct table, correct quarters identified
- 0.7-0.8: Found ESF data but some ambiguity in quarter identification
- 0.4-0.6: Found relevant financial data but wrong section or periods
- 0.0-0.3: Failed to locate ESF data

=== Part 3: Answer Presentation (weight 25%) ===
- Reported with proper precision (3 decimal places: 4.815%)
- Stated this is an absolute (unsigned) percent change
- Provided context (ESF, total assets, Q2→Q3 2022)

Part 3 scoring:
- 0.9-1.0: Correct precision, noted absolute change, full context
- 0.6-0.8: Correct precision but minimal context
- 0.3-0.5: Answer given but wrong precision or missing context
- 0.0-0.2: No clear answer

Output the final weighted score: score = 0.40×Part1 + 0.35×Part2 + 0.25×Part3"""

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

        # 3) Computation methodology (0.35) — LLM Judge
        completion += 0.35 * self._call_judge(
            judge, task.prompt.text, conversation, actions_summary,
            self._METHODOLOGY_RUBRIC,
            fallback=self._fb_methodology(all_text),
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

    def _fb_methodology(self, all_text: str) -> float:
        """Fallback: QoQ methodology via keyword/pattern matching."""
        score = 0.0
        all_lower = all_text.lower()

        # QoQ computation (0.40)
        has_qoq = "qoq" in all_lower or "quarter" in all_lower
        has_percent = "percent" in all_lower or "%" in all_text
        has_change = any(kw in all_lower for kw in ["change", "difference", "growth"])
        has_formula = any(kw in all_lower for kw in ["divide", "ratio", "/"])
        method_signals = sum([has_qoq, has_percent, has_change, has_formula])
        score += 0.40 * min(method_signals / 2.5, 1.0)

        # ESF data identification (0.35)
        has_esf = "esf" in all_lower or "exchange stabilization" in all_lower
        has_assets = "asset" in all_lower
        has_june = "june" in all_lower or "q2" in all_lower
        has_sept = "september" in all_lower or "q3" in all_lower
        data_signals = sum([has_esf, has_assets, has_june, has_sept])
        score += 0.35 * min(data_signals / 2.5, 1.0)

        # Answer presentation (0.25)
        has_4815 = "4.815" in all_text
        has_precision = bool(re.search(r"\d+\.\d{3}", all_text))
        pres_signals = sum([has_4815, has_precision])
        score += 0.25 * min(pres_signals / 1.5, 1.0)

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
