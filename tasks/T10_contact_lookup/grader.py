"""T05en_contact_lookup grader — English variant."""

import re

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T09zh_contact_lookup")


class ContactLookupGraderEN(_Base):
    """English variant — overrides language-specific class attributes and rubrics."""

    CORRECT_CONTACT = {
        "id": "c_001",
        "name": "David Zhang",
        "department": "Engineering",
        "title": "Senior Engineer",
        "email": "dzhang@company.com",
        "phone": "138-0001-1001",
    }

    HOMOPHONE_DISTRACTOR = {
        "id": "c_007",
        "name": "David Chang",
        "department": "Engineering",
        "title": "Junior Engineer",
        "email": "dchang@company.com",
        "phone": "133-0007-7007",
    }

    SIMILAR_NAMES = ["David Zhang", "David Chang", "Dave Zhang", "D. Zhang"]

    RECOMMENDATION_KEYWORDS = ["recommend", "suggest", "should be", "likely"]
    DISTINCTION_KEYWORDS = ["difference", "distinguish", "unlike", "note", "rather than"]
    SENIOR_TITLE_KW = "Senior"
    JUNIOR_TITLE_KW = "Junior"
    LOCATION_KEYWORDS = ["Beijing", "Floor 5", "Building A", "5th Floor"]

    COMM_ENTITIES_EXTRA = ["Dave Zhang", "D. Zhang"]

    # ------------------------------------------------------------------ #
    # LLM Judge rubric (English)
    # ------------------------------------------------------------------ #

    _DISAMBIGUATION_RUBRIC = """\
Evaluate the agent's disambiguation quality when facing multiple similar-name \
search results.

Context: The user asked for "David Zhang in Engineering". Search results include:
- David Zhang (c_001, Engineering, Senior Engineer) — correct target
- David Chang (c_007, Engineering, Junior Engineer) — similar-name distractor, \
same department
- Dave Zhang, D. Zhang, etc. — other similar names

Key disambiguation criteria:
1. Did the agent list multiple search results to show the user there are \
similar names?
2. Did the agent clearly distinguish David Zhang (Senior Engineer) from \
David Chang (Junior Engineer) — both in Engineering?
3. Did the agent clearly recommend the correct David Zhang (c_001), not \
David Chang (c_007)?
4. Did the agent use department filtering to proactively narrow results?
5. Did the agent provide additional useful info like location?

Important: If the agent recommended the wrong contact (David Chang/c_007 \
instead of David Zhang/c_001), this should be heavily penalized.

Scoring:
- 0.9-1.0: Listed multiple results, clearly distinguished Zhang vs Chang, \
correct recommendation with reasoning
- 0.7-0.8: Correct recommendation but distinction explanation not thorough
- 0.4-0.6: Mentioned multiple results but disambiguation unclear
- 0.1-0.3: No effective disambiguation, or recommended wrong contact
- 0.0: No disambiguation analysis at all"""

    def _deterministic_communication(self, final_text: str, names_mentioned: int = 0) -> float:
        """English deterministic communication scoring."""
        has_structure = bool(re.search(r"[-*]\s|^\d+\.|##|\|.*\|", final_text, re.MULTILINE))
        has_comparison = names_mentioned >= 3
        has_recommendation = any(kw in final_text for kw in self.RECOMMENDATION_KEYWORDS)
        has_reasoning = self.CORRECT_CONTACT["department"] in final_text and \
            ("API" in final_text or "engineer" in final_text.lower())

        format_score = 0.0
        if has_structure:
            format_score += 0.25
        if has_comparison:
            format_score += 0.25
        if has_recommendation:
            format_score += 0.25
        if has_reasoning:
            format_score += 0.25

        tool_entities = [
            self.CORRECT_CONTACT["email"],
            self.CORRECT_CONTACT["phone"],
            self.CORRECT_CONTACT["title"],
            self.CORRECT_CONTACT["name"],
            self.HOMOPHONE_DISTRACTOR["name"],
        ] + self.COMM_ENTITIES_EXTRA
        return self.compute_communication_substance(
            final_text, tool_entities, format_score
        )
