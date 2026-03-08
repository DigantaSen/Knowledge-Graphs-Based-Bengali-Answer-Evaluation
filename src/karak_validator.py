"""
karak_validator.py - Semantic-role (Karak / কারক) validation: detects agent-object swaps.
"""

class KarakValidator:
    """Validates semantic roles to detect agent-object swaps in student answers."""

    # ── role extraction ───────────────────────────────────────────────────────

    @staticmethod
    def extract_roles(parsed_sentence: list) -> dict:
        roles = {
            "agent": set(),
            "object": set(),
            "location_time": set(),
            "instrument": set(),
        }
        for word in parsed_sentence:
            deprel = word["deprel"]
            lemma = word["lemma"] if word["lemma"] else word["text"]
            if "subj" in deprel:
                roles["agent"].add(lemma)
            elif "obj" in deprel:
                roles["object"].add(lemma)
            elif deprel in ("obl:loc", "obl:tmod"):
                roles["location_time"].add(lemma)
            elif deprel in ("obl:ins", "obl"):
                roles["instrument"].add(lemma)
        return roles

    # ── penalty computation ───────────────────────────────────────────────────

    @staticmethod
    def _swap_penalty(ref_roles: dict, stu_roles: dict):
        penalty = 0.0
        explanations = []

        ref_all = ref_roles["agent"] | ref_roles["object"]
        stu_all = stu_roles["agent"] | stu_roles["object"]

        if ref_all == stu_all:
            return 0.0, []

        for agent in ref_roles["agent"]:
            if agent in stu_roles["object"] and agent not in stu_roles["agent"]:
                penalty += 0.5
                explanations.append(
                    f"Role swap: '{agent}' was AGENT in reference, but OBJECT in student answer."
                )

        for obj in ref_roles["object"]:
            if obj in stu_roles["agent"] and obj not in stu_roles["object"]:
                penalty += 0.5
                explanations.append(
                    f"Role swap: '{obj}' was OBJECT in reference, but AGENT in student answer."
                )

        return min(penalty, 1.0), explanations

    # ── public API ────────────────────────────────────────────────────────────

    def validate(self, ref_parsed: list, stu_parsed: list):
        """Return ``(penalty, explanations)`` where penalty is in [0.0, 1.0]."""
        ref_roles = {"agent": set(), "object": set(), "location_time": set(), "instrument": set()}
        stu_roles = {"agent": set(), "object": set(), "location_time": set(), "instrument": set()}

        for sent in ref_parsed:
            r = self.extract_roles(sent)
            for k, v in r.items():
                ref_roles[k].update(v)

        for sent in stu_parsed:
            r = self.extract_roles(sent)
            for k, v in r.items():
                stu_roles[k].update(v)

        penalty, explanations = self._swap_penalty(ref_roles, stu_roles)
        missing_locs = ref_roles["location_time"] - stu_roles["location_time"]
        if missing_locs:
            penalty += 0.1 * len(missing_locs)

        return min(penalty, 1.0), explanations
