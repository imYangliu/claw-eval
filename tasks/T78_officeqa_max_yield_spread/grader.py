"""T52_officeqa_max_yield_spread grader — maximum yield spread 1960-1969."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "031969"
TOLERANCE = 0.05
COMM_ENTITIES = ["yield", "spread", "corporate", "Treasury", "bonds", "1960", "1969", "Aa"]


class OfficeQAMaxYieldSpreadGrader(AbstractGrader):
    """Grader for T52: maximum yield spread between corporate Aa and Treasury bonds.

    The agent must OCR a Treasury Bulletin (June 1970), locate the monthly yield
    spread data for 1960-1969, compare all months to find the maximum, and report
    the result in MMYYYY format (answer: 031969 = March 1969).

    This is a multi-step reasoning task: OCR → locate table → compute spreads
    (or read them) → find max → format as MMYYYY.

    Scoring: rule-based for numerical accuracy and OCR tool usage;
    LLM judge for analytical reasoning quality.
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _ANALYSIS_RUBRIC = """\
Evaluate the agent's analytical reasoning in finding the month with maximum \
yield spread between US corporate Aa bonds and US Treasury bonds during 1960-1969.
The correct answer is 031969 (March 1969 in MMYYYY format).
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Yield Spread Computation & Comparison (weight 45%) ===
The agent needed to:
- Identify or compute yield spreads for each month across 1960-1969
- Compare spreads across the full 10-year period to find the maximum
- Show its work: which months/years had high spreads, how it determined the max

Part 1 scoring:
- 0.9-1.0: Showed spread values for multiple months/years, clearly demonstrated \
March 1969 had the maximum with supporting data
- 0.7-0.8: Identified March 1969 as max with some supporting data but incomplete comparison
- 0.5-0.6: Found a high-spread period (late 1960s) but didn't precisely identify March 1969
- 0.2-0.4: Attempted spread calculation but major errors or incomplete
- 0.0-0.1: No spread computation or comparison attempted

=== Part 2: Data Source & Table Identification (weight 25%) ===
- Referenced the Treasury Bulletin (June 1970) as source
- Identified the correct table with corporate Aa bond yields AND Treasury bond yields
- Correctly understood that spread = corporate yield - Treasury yield (or similar)

Part 2 scoring:
- 0.9-1.0: Clear source reference, correct table identified, spread definition understood
- 0.6-0.8: Source mentioned, table roughly identified
- 0.3-0.5: Gave answer without clear table reference
- 0.0-0.2: No source or table identification

=== Part 3: Answer Format & Presentation (weight 30%) ===
- Correctly formatted answer as MMYYYY integer (031969 for March 1969)
- Explained what the number represents (month=03, year=1969)
- Stated the actual spread value at that peak month

Part 3 scoring:
- 0.9-1.0: Correct MMYYYY format, explained the encoding, stated the peak spread value
- 0.6-0.8: Correct format but didn't explain encoding or omitted spread value
- 0.3-0.5: Identified March 1969 but wrong format (e.g., "March 1969" instead of 031969)
- 0.0-0.2: No clear answer or completely wrong format

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

        # 3) Analytical reasoning quality (0.35) — LLM Judge
        completion += 0.35 * self._call_judge(
            judge, task.prompt.text, conversation, actions_summary,
            self._ANALYSIS_RUBRIC,
            fallback=self._fb_analysis(all_text),
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

    def _fb_analysis(self, all_text: str) -> float:
        """Fallback: analytical reasoning via keyword/pattern matching."""
        score = 0.0
        all_lower = all_text.lower()

        # Spread computation & comparison (0.45)
        has_spread = "spread" in all_lower
        has_yield = "yield" in all_lower
        has_corporate = "corporate" in all_lower or "aa" in all_lower
        has_treasury = "treasury" in all_lower
        has_comparison = any(kw in all_lower for kw in [
            "maximum", "max", "highest", "largest", "peak",
        ])
        has_1969 = "1969" in all_text
        has_march = "march" in all_lower or "mar" in all_lower

        compute_signals = sum([has_spread, has_yield, has_corporate, has_treasury,
                               has_comparison, has_1969, has_march])
        score += 0.45 * min(compute_signals / 4, 1.0)

        # Data source (0.25)
        source_kw = ["treasury", "bulletin", "1970", "ocr", "table"]
        source_count = sum(1 for kw in source_kw if kw in all_lower)
        score += 0.25 * min(source_count / 2, 1.0)

        # Answer format (0.30)
        has_031969 = "031969" in all_text
        has_mmyyyy = "mmyyyy" in all_lower or "mm" in all_lower
        has_format_explain = has_march and has_1969
        fmt_signals = sum([has_031969, has_mmyyyy or has_format_explain])
        score += 0.30 * min(fmt_signals / 1.5, 1.0)

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
