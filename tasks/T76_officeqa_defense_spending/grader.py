"""T50_officeqa_defense_spending grader — U.S. national defense expenditures 1940."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "2,602"
TOLERANCE = 0.05
DOMAIN_KEYWORDS = ["defense", "expenditure", "national", "1940", "million"]
COMM_ENTITIES = ["defense", "expenditure", "1940", "2,602", "million", "Treasury"]


class OfficeQADefenseSpendingGrader(AbstractGrader):
    """Grader for T50: U.S. national defense expenditures (1940).

    The agent must use OCR to extract text from a Treasury Bulletin scan,
    locate the defense expenditures table, and report 2,602 million dollars.

    Scoring: rule-based for numerical accuracy and OCR tool usage;
    LLM judge for answer explanation quality.
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _ANSWER_QUALITY_RUBRIC = """\
Evaluate the quality of the agent's answer explanation for finding U.S. national \
defense expenditures in 1940 from a Treasury Bulletin.
Score each of the two parts separately, then compute a weighted final score (0-1).

=== Part 1: Data Source & Extraction Process (weight 50%) ===
The agent should demonstrate it correctly processed the OCR output:
- Referenced the Treasury Bulletin as the data source
- Identified the correct table/section containing defense expenditures
- Showed how it located the 1940 calendar year data
- Distinguished between fiscal year and calendar year if relevant

Part 1 scoring:
- 0.9-1.0: Clearly described data source and extraction process, referenced \
specific table/section
- 0.6-0.8: Mentioned data source but extraction process unclear
- 0.3-0.5: Gave an answer without explaining how it was found
- 0.0-0.2: No reference to data source or extraction method

=== Part 2: Answer Presentation & Context (weight 50%) ===
The agent should present the answer clearly with appropriate context:
- Stated the answer with correct units (millions of dollars)
- Provided context (e.g., "national defense" category, calendar year 1940)
- Noted any caveats or data quality issues from OCR

Part 2 scoring:
- 0.9-1.0: Clear answer with units, context, and any relevant caveats
- 0.6-0.8: Answer with units but minimal context
- 0.3-0.5: Just a number without units or context
- 0.0-0.2: No clear answer presented

Output the final weighted score: score = 0.50×Part1 + 0.50×Part2"""

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
        scores.safety = 1.0  # No safety gate for QA tasks

        final_text = self._get_final_assistant_text(messages)
        all_text = self._get_all_assistant_text(messages)
        conversation = self.format_conversation(messages)
        actions_summary = self.summarize_actions(audit_data)

        # ── Completion ──────────────────────────────────────────────
        completion = 0.0

        # 1) Numerical accuracy (0.70) — rule-based
        try:
            answer_text = extract_final_answer(all_text) if all_text else ""
            if answer_text:
                numerical_score = score_answer(GROUND_TRUTH, answer_text, TOLERANCE)
                completion += 0.70 * numerical_score
        except Exception:
            pass

        # 2) OCR tool usage (0.15) — rule-based
        ocr_calls = [d for d in dispatches if d.tool_name == "ocr_extract_text"]
        if ocr_calls:
            completion += 0.15

        # 3) Answer explanation quality (0.15) — LLM Judge
        completion += 0.15 * self._call_judge(
            judge, task.prompt.text, conversation, actions_summary,
            self._ANSWER_QUALITY_RUBRIC,
            fallback=self._fb_answer_quality(all_text),
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

    def _fb_answer_quality(self, all_text: str) -> float:
        """Fallback: answer quality via keyword matching."""
        score = 0.0
        all_lower = all_text.lower()

        # Data source references (0.50)
        source_kw = ["treasury", "bulletin", "ocr", "table", "extract"]
        source_count = sum(1 for kw in source_kw if kw in all_lower)
        score += 0.50 * min(source_count / 2, 1.0)

        # Domain coverage (0.50)
        kw_count = sum(1 for kw in DOMAIN_KEYWORDS if kw.lower() in all_lower)
        score += 0.50 * min(kw_count / 3, 1.0)

        return min(score, 1.0)

    # ================================================================== #
    # Communication fallback
    # ================================================================== #

    def _deterministic_communication(self, final_text: str) -> float:
        has_headers = bool(re.search(r"##|###|\*\*.*\*\*", final_text))
        has_bullets = bool(
            re.search(r"[-*]\s|^\d+\.", final_text, re.MULTILINE)
        )
        has_sections = final_text.count("##") >= 2 or final_text.count("**") >= 3

        format_score = 0.0
        if has_headers:
            format_score += 0.35
        if has_bullets:
            format_score += 0.30
        if has_sections:
            format_score += 0.35
        format_score = min(format_score, 1.0)

        return self.compute_communication_substance(
            final_text, COMM_ENTITIES, format_score
        )
