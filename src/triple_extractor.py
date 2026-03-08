"""
triple_extractor.py - Hybrid (S, R, O) triple extraction for Bengali.

Primary:  dependency-parse based extraction (Stanza + SOV heuristics)
Fallback: rule-based extractor for constructions the dep-parser misses
          (passive voice, nominal predicates, negation, postpositional objects)

Note: The mT5 generative fallback present in earlier versions was removed.
See class docstring for the full rationale.
"""


class TripleExtractor:
    """Hybrid (dependency-parse + rule-based) triple extractor for Bengali."""

    # Postposition → semantic role
    POSTPOSITIONS = {
        "কে": "object", "কেই": "object",
        "র": "possessive", "এর": "possessive",
        "তে": "location", "তেই": "location",
        "থেকে": "source", "হতে": "source",
        "দিয়ে": "instrument", "দ্বারা": "instrument",
        "জন্য": "purpose", "জন্যে": "purpose",
        "সাথে": "companion", "সঙ্গে": "companion",
    }

    COPULAS = {
        "হল", "হলো", "হয়", "হচ্ছে", "ছিল", "ছিলেন", "ছিলাম",
        "হবে", "হবেন", "হওয়া", "থাকে", "থাকেন", "আছে", "আছেন",
        "হলেন", "হন", "হয়েছে", "হয়েছিল",
    }

    NEGATIONS = {"না", "নয়", "নি", "নে", "নেই", "না-ই", "নাই"}

    def __init__(self, use_generative_fallback: bool = False):
        if use_generative_fallback:
            print("  ℹ  mT5 generative fallback disabled by design (see class docstring).")
            print("     Using enhanced rule-based extractor instead.")

    # ── dependency-parse extraction ───────────────────────────────────────────

    def _dep_extract(self, parsed_sentences: list) -> list:
        triples = []
        for sentence in parsed_sentences:
            subject = obj = relation = root_id = None
            aux_parts = []
            neg_flag = False

            for word in sentence:
                lemma = word["lemma"] if word["lemma"] else word["text"]
                deprel = word["deprel"]

                if "subj" in deprel:
                    subject = lemma
                elif deprel == "root":
                    relation = lemma
                    root_id = word["id"]
                elif "obj" in deprel or deprel in (
                    "obl", "obl:loc", "obl:tmod", "obl:ins", "obl:arg"
                ):
                    obj = lemma
                elif deprel in ("aux", "compound:lvc", "cop"):
                    aux_parts.append(lemma)
                elif deprel == "advmod" and lemma in self.NEGATIONS:
                    neg_flag = True

            if relation and root_id is not None and aux_parts:
                relation = f"{relation} {' '.join(aux_parts)}"

            if neg_flag and relation:
                relation = f"NOT({relation})"

            if subject and relation and obj:
                triples.append((subject, relation, obj))
            elif subject and relation:
                triples.append((subject, relation, "[NONE]"))

        return triples

    # ── rule-based fallback extraction ───────────────────────────────────────

    def _rule_based_extract(self, text: str) -> list:
        triples = []
        sentences = [s.strip() for s in text.replace("।", "।\n").split("\n") if s.strip()]

        for sent in sentences:
            words = sent.replace("।", "").split()
            if len(words) < 2:
                continue

            for i, w in enumerate(words):
                if w in self.COPULAS and i > 0:
                    subject = words[0]
                    obj_part = " ".join(words[1:i])
                    pred = " ".join(words[i:])
                    if obj_part:
                        triples.append((subject, pred, obj_part))
                    break

            subject_cand = words[0] if words else None
            verb_cand = obj_cand = None

            for w in words[1:]:
                for pp, role in self.POSTPOSITIONS.items():
                    if w.endswith(pp) and len(w) > len(pp) + 1:
                        stripped = w[: -len(pp)]
                        if role in ("object", "possessive"):
                            obj_cand = stripped
                        break
                if w not in self.NEGATIONS:
                    verb_cand = w

            neg_flag = any(w in self.NEGATIONS for w in words)
            relation = f"NOT({verb_cand})" if neg_flag and verb_cand else verb_cand

            if subject_cand and relation and obj_cand:
                if (subject_cand, relation, obj_cand) not in triples:
                    triples.append((subject_cand, relation, obj_cand))

            if not obj_cand and len(words) >= 3:
                s, v = words[0], words[-1]
                o = " ".join(words[1:-1])
                if len(o) > 1:
                    if neg_flag:
                        v = f"NOT({v})"
                    if (s, v, o) not in triples:
                        triples.append((s, v, o))

        return triples

    # ── public API ────────────────────────────────────────────────────────────

    def extract_triples(self, parsed_sentences: list, raw_text: str = "") -> list:
        """Extract (subject, relation, object) triples.

        Uses dependency-parse first; falls back to rule-based only when
        the dep-parse yields no triples.
        """
        triples = self._dep_extract(parsed_sentences)
        if not triples and raw_text:
            triples = self._rule_based_extract(raw_text)
        return triples
