"""T53_officeqa_zipf_exponent grader — Zipf exponent for unemployment insurance tax receipts 2020."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "1.172"
TOLERANCE = 0.05
COMM_ENTITIES = ["Zipf", "exponent", "unemployment", "insurance", "2020", "states", "regression"]


class OfficeQAZipfExponentGrader(AbstractGrader):
    """Grader for T53: Zipf exponent for state unemployment insurance tax receipts.

    The agent must OCR a Treasury Bulletin (December 2020), extract unemployment
    insurance tax receipt data for all 50 states (excluding DC), perform log-log
    regression to compute the Zipf exponent, and report 1.172 (3 decimal places).

    This is a multi-step task: OCR → extract state-level data → rank by amount →
    log-log regression → compute exponent → round to 3 decimals.

    Scoring: rule-based for numerical accuracy and OCR tool usage;
    LLM judge for statistical methodology quality.
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _METHODOLOGY_RUBRIC = """\
Evaluate the agent's statistical methodology in computing the Zipf exponent for \
state unemployment insurance tax receipts from the 2020 Treasury Bulletin.
The correct answer is 1.172 (Zipf exponent rounded to 3 decimal places).
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Zipf Law Understanding & Log-Log Regression (weight 45%) ===
The agent needed to:
- Understand that the Zipf exponent is the slope of a log-log regression \
(log(rank) vs log(value))
- Apply linear regression / least-squares fitting on the log-transformed data
- Show the regression equation or at least describe the fitting process
- Report the exponent (slope) with appropriate precision

Part 1 scoring:
- 0.9-1.0: Clear description of log-log regression methodology, showed fitting \
process, correctly identified exponent as the slope
- 0.7-0.8: Used log-log regression but explanation was incomplete
- 0.5-0.6: Mentioned Zipf's law but methodology was vague or partially incorrect
- 0.2-0.4: Attempted some computation but didn't use proper log-log regression
- 0.0-0.1: No statistical methodology described

=== Part 2: Data Extraction & Preparation (weight 30%) ===
- Extracted unemployment insurance tax receipt data for all 50 states (excluding DC)
- Correctly identified the data source (Treasury Bulletin December 2020)
- Ranked states by tax receipt amount (descending) before regression
- Handled data quality issues from OCR (if any)

Part 2 scoring:
- 0.9-1.0: All 50 states extracted, DC excluded, data correctly ranked
- 0.7-0.8: Most states extracted with correct ranking, minor omissions
- 0.5-0.6: Substantial data extracted but incomplete or ranking unclear
- 0.2-0.4: Some data extracted but major gaps or errors
- 0.0-0.1: No meaningful data extraction

=== Part 3: Answer Presentation & Precision (weight 25%) ===
- Reported the Zipf exponent with 3 decimal places (1.172)
- Stated units/interpretation (dimensionless exponent, slope of log-log fit)
- Provided context about what the exponent means for the distribution

Part 3 scoring:
- 0.9-1.0: Correct precision (3 decimals), interpreted the result, explained context
- 0.6-0.8: Correct precision but minimal interpretation
- 0.3-0.5: Result reported but wrong precision or no interpretation
- 0.0-0.2: No clear answer presented

Output the final weighted score: score = 0.45×Part1 + 0.30×Part2 + 0.25×Part3"""

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

        # 3) Statistical methodology quality (0.35) — LLM Judge
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
        """Fallback: statistical methodology via keyword/pattern matching."""
        score = 0.0
        all_lower = all_text.lower()

        # Zipf law & regression methodology (0.45)
        has_zipf = "zipf" in all_lower
        has_log = "log" in all_lower
        has_regression = any(kw in all_lower for kw in [
            "regression", "least squares", "fit", "slope", "线性回归",
        ])
        has_exponent = any(kw in all_lower for kw in [
            "exponent", "指数", "slope", "斜率",
        ])
        has_rank = "rank" in all_lower or "排名" in all_lower
        method_signals = sum([has_zipf, has_log, has_regression, has_exponent, has_rank])
        score += 0.45 * min(method_signals / 3, 1.0)

        # Data extraction (0.30)
        has_states = "state" in all_lower or "州" in all_lower
        has_unemployment = "unemployment" in all_lower or "失业" in all_lower
        has_insurance = "insurance" in all_lower or "保险" in all_lower
        has_50 = "50" in all_text
        has_dc = "dc" in all_lower or "columbia" in all_lower
        data_signals = sum([has_states, has_unemployment, has_insurance, has_50, has_dc])
        score += 0.30 * min(data_signals / 3, 1.0)

        # Answer presentation (0.25)
        has_1172 = "1.172" in all_text
        has_decimal = bool(re.search(r"1\.\d{3}", all_text))
        has_interpret = any(kw in all_lower for kw in [
            "dimensionless", "slope", "power law", "幂律",
        ])
        pres_signals = sum([has_1172, has_decimal, has_interpret])
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
