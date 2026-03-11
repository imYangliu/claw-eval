"""T59_officeqa_army_expenditures grader — cross-document Army expenditures comparison."""

from __future__ import annotations

import logging
import re
from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.officeqa_reward import extract_final_answer, score_answer
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage

log = logging.getLogger(__name__)

GROUND_TRUTH = "6,244"
TOLERANCE = 0.05
COMM_ENTITIES = [
    "Army", "expenditure", "1940", "1947", "6,244", "million", "increase",
    "Department",
]


class OfficeQAArmyExpendituresGrader(AbstractGrader):
    """Grader for T59: multi-source Army expenditures comparison.

    This is the only multi-source OfficeQA task — the agent must extract data from
    two different Treasury Bulletins (April 1948 and December 1952), find Department
    of the Army expenditures for FY1940 and FY1947, and compute the increase
    of 6,244 million dollars.

    The multi-source nature makes cross-document reasoning critical: the agent must
    correctly identify which document contains which fiscal year's data.

    Scoring: rule-based for numerical accuracy and OCR tool usage;
    LLM judge for cross-document reasoning quality (weighted higher at 0.40).
    """

    # ------------------------------------------------------------------ #
    # LLM Judge rubric
    # ------------------------------------------------------------------ #

    _CROSS_DOC_RUBRIC = """\
Evaluate the agent's cross-document reasoning in computing the increase in U.S. \
Department of the Army expenditures from FY1940 to FY1947, using data from two \
different Treasury Bulletins (April 1948 and December 1952).
The correct answer is 6,244 million dollars.
Score each of the three parts separately, then compute a weighted final score (0-1).

=== Part 1: Cross-Document Data Integration (weight 40%) ===
This is a MULTI-SOURCE task. The agent needed to:
- Recognize that data must come from two separate documents
- Identify which bulletin contains FY1940 data and which contains FY1947 data
- Extract the correct Army expenditure figures from each document
- Reconcile any differences in table format or naming conventions between bulletins

Part 1 scoring:
- 0.9-1.0: Explicitly referenced both documents, correctly attributed data to each, \
showed awareness of cross-document nature
- 0.7-0.8: Used data from both documents but attribution unclear
- 0.4-0.6: Found some data but didn't clearly distinguish between documents
- 0.2-0.3: Only used one document
- 0.0-0.1: Failed to extract data from either document

=== Part 2: Expenditure Comparison & Computation (weight 35%) ===
- Correctly identified Department of the Army (not total defense) expenditures
- Extracted FY1940 and FY1947 values
- Computed the increase (FY1947 - FY1940)
- Showed computation steps with actual figures

Part 2 scoring:
- 0.9-1.0: Correct department, both values shown, computation clear
- 0.7-0.8: Correct computation but department identification ambiguous
- 0.4-0.6: Found relevant data but computation errors
- 0.0-0.3: Failed to identify or compare the expenditure figures

=== Part 3: Answer Presentation (weight 25%) ===
- Stated the increase amount with units (millions of dollars)
- Provided context (Department of the Army, FY1940 vs FY1947)
- Referenced both source documents

Part 3 scoring:
- 0.9-1.0: Clear answer with units, context, and both sources referenced
- 0.6-0.8: Answer with units but missing source references
- 0.3-0.5: Just a number without proper context
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

        # 1) Numerical accuracy (0.50) — rule-based
        #    Slightly lower weight because cross-doc reasoning is the key challenge
        try:
            answer_text = extract_final_answer(all_text) if all_text else ""
            if answer_text:
                completion += 0.50 * score_answer(GROUND_TRUTH, answer_text, TOLERANCE)
        except Exception:
            pass

        # 2) OCR tool usage (0.10) — rule-based
        if any(d.tool_name == "ocr_extract_text" for d in dispatches):
            completion += 0.10

        # 3) Cross-document reasoning quality (0.40) — LLM Judge
        #    Higher weight than other OfficeQA tasks due to multi-source nature
        completion += 0.40 * self._call_judge(
            judge, task.prompt.text, conversation, actions_summary,
            self._CROSS_DOC_RUBRIC,
            fallback=self._fb_cross_doc(all_text),
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

    def _fb_cross_doc(self, all_text: str) -> float:
        """Fallback: cross-document reasoning via keyword/pattern matching."""
        score = 0.0
        all_lower = all_text.lower()

        # Cross-document data integration (0.40)
        has_1948_doc = "1948" in all_text
        has_1952_doc = "1952" in all_text
        has_both_docs = has_1948_doc and has_1952_doc
        has_bulletin = "bulletin" in all_lower
        has_doc_ref = any(kw in all_lower for kw in [
            "document 1", "document 2", "first bulletin", "second bulletin",
            "both document", "both bulletin", "two document", "two bulletin",
        ])
        if has_both_docs and has_doc_ref:
            score += 0.40
        elif has_both_docs:
            score += 0.30
        elif has_1948_doc or has_1952_doc:
            score += 0.15
        elif has_bulletin:
            score += 0.05

        # Expenditure comparison (0.35)
        has_army = "army" in all_lower
        has_1940 = "1940" in all_text
        has_1947 = "1947" in all_text
        has_increase = any(kw in all_lower for kw in ["increase", "difference", "change", "grew"])
        has_6244 = "6244" in all_text.replace(",", "")
        compute_signals = sum([has_army, has_1940, has_1947, has_increase, has_6244])
        score += 0.35 * min(compute_signals / 3, 1.0)

        # Answer presentation (0.25)
        has_million = "million" in all_lower
        has_dept = "department" in all_lower
        pres_signals = sum([has_6244, has_million, has_dept])
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
        return self.compute_communication_substance(
            final_text, COMM_ENTITIES, min(fmt, 1.0)
        )
