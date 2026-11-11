from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

from core.metadata import extract_metadata
from orthography_change.models.rule import OrthographyRule
from preprocessing import preprocess


class OrthographyAnalyzer:
    """
    Analyse un fichier CoNLL-U et compte les occurrences des variantes orthographiques définies par les règles chargées.
    """

    def __init__(
        self,
        old_index: dict[str, OrthographyRule],
        new_index: dict[str, OrthographyRule],
    ):
        self.old_index = old_index
        self.new_index = new_index

    def analyze(self, path: Path) -> dict:
        """
        Analyse un fichier CoNLL-U.
        """
        counts, _ = preprocess(path)
        metadata = extract_metadata(path)
        return {
            **metadata,
            "file": path,
            "counts": self._match_rules(counts),
        }

    def analyze_from_cache(self, record: dict) -> dict:
        """Analyze one preprocessed (cached) issue record.

        Args:
            record: One entry from preprocessing.load_preprocessed_corpus, with keys "journal", "year", "date", "file", "counts" (nested dict mapping part_of_speech -> word -> feats -> count), and "total_words".

        Returns:
            Dict with keys "journal", "year", "date", "file" (Path), and "counts" (dict mapping matched rule.old_form/rule.new_form to their summed occurrence counts).
        """

        pos_word_feats_counts: Dict[Tuple[str, str, str], int] = {}
        for pos, word_to_feats in record["counts"].items():
            for word, feats_to_count in word_to_feats.items():
                for feats, count in feats_to_count.items():
                    pos_word_feats_counts[(pos, word, feats)] = count
        return {
            "journal": record["journal"],
            "year": record["year"],
            "date": record["date"],
            "file": Path(record["file"]),
            "counts": self._match_rules(pos_word_feats_counts),
        }

    def _match_rules(
        self,
        pos_word_feats_counts: Dict[Tuple[str, str, str], int],
    ) -> dict[str, int]:
        """Match (pos, word, feats) counts against loaded rules.

        Args:
            pos_word_feats_counts: Mapping of (part_of_speech,
                lowercased word form, canonicalized feats string) to
                its occurrence count.
        Returns:
            Dict mapping matched rule.old_form/rule.new_form to
                their summed occurrence counts.
        """

        counts = Counter()
        for (upos, form, feats), count in pos_word_feats_counts.items():
            # ancienne forme
            rule = self.old_index.get(form)
            if rule is not None and rule.matches_morphology(
                upos, feats
            ):
                counts[rule.old_form] += count
                continue
            # nouvelle forme
            rule = self.new_index.get(form)
            if rule is not None and rule.matches_morphology(
                upos, feats
            ):
                counts[rule.new_form] += count
        return dict(counts)
