import json
from pathlib import Path
from typing import Dict, Tuple

from conllu import parse_incr
from joblib import Parallel, delayed
from tqdm import tqdm

from config import RULES_FILE
from core.filesystem import FileSystemScanner
from core.metadata import extract_metadata
from orthography_change.rules.loader import load_rules


def _feats_to_str(feats: dict | None) -> str:
    """Canonicalize a token's parsed FEATS into a sorted string.

    Args:
        feats: Dictionary of morphological feature key/value pairs
            (as parsed by the conllu library), or None when the
            CoNLL-U FEATS column is "_".

    Returns:
        String of "Key=Value" pairs, sorted by key and joined with
            "|" (e.g. "Number=Sing|VerbForm=Part"), or "" if feats
            is None or empty.
    """

    if not feats:
        return ""

    return "|".join(f"{key}={value}" for key, value in sorted(
        feats.items()
    ))


def preprocess(
    issue_conllu_path: Path,
    allowed_forms: set[str] | None = None,
) -> Tuple[Dict[Tuple[str, str, str], int], int]:
    """Count (part_of_speech, word, feats) triples in a CONLLU file.

    Each text file pertains to an issue of a newspaper, published at a
    specific date, that contains multiple articles. This function
    does not handle bookkeeping with respect to issue ID, newspaper
    ID, publishing date, and possibly other data. If the count is
    zero, no triple is generated.

    Intended for use with 38,000 CONLLU files. Should be applied to
    an entire corpus in parallel via preprocess_corpus.

    Args:
        issue_conllu_path: Path corresponding to the CONLLU file.
        allowed_forms: Optional set of lowercased word forms to
            restrict counts to; tokens whose lowercased form is not
            in this set are skipped when building counts, but are
            still counted toward total_words. None (default) keeps
            every token.

    Returns:
        Tuple of:
            - Dictionary mapping (part_of_speech, word, feats)
                tuples to their frequency counts.
            - Total number of word tokens in the file, i.e. the
                denominator for later relative-frequency
                computations. Always reflects every token,
                regardless of allowed_forms.
    """

    counts: Dict[Tuple[str, str, str], int] = {}
    total_words = 0

    with issue_conllu_path.open("r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            for token in sentence:
                # ignore multiword / empty nodes
                if not isinstance(token["id"], int):
                    continue
                total_words += 1
                form = token["form"].lower()
                if (allowed_forms is not None and form not in allowed_forms):
                    continue

                key = (token["upos"], form, _feats_to_str(token["feats"]))
                counts[key] = counts.get(key, 0) + 1

    return counts, total_words


def _load_allowed_forms(rules_path: Path) -> set[str]:
    """Load the set of word forms referenced by orthography rules.

    Args:
        rules_path: Path to the rules CSV file, passed to
            orthography_change.rules.loader.load_rules.

    Returns:
        Set of lowercased old_form/new_form values across every
            rule in rules_path.
    """

    forms: set[str] = set()

    for rule in load_rules(rules_path):
        forms.add(rule.old_form.lower())
        forms.add(rule.new_form.lower())

    return forms


def _preprocess_issue_record(
    issue_conllu_path: Path,
    allowed_forms: set[str] | None = None,
) -> dict:
    """Preprocess one CONLLU file into a nested per-issue JSON record.

    Module-level so it can be pickled for joblib's parallel dispatch.
    Builds a {pos: {word: {feats: count}}} nested shape - an issue
    with nothing to keep just gets an empty "counts" dict.

    Args:
        issue_conllu_path: Path corresponding to the CONLLU file.
        allowed_forms: Optional set of lowercased word forms to
            restrict counts to; forwarded to preprocess(). None
            (default) keeps every token.

    Returns:
        Dict with keys "journal" (str), "year" (int), "date" (str or
            None), "file" (str), "counts" (dict mapping
            part_of_speech to a dict mapping word to a dict mapping
            feats to count), and "total_words" (int).
    """

    counts, total_words = preprocess(issue_conllu_path, allowed_forms)
    metadata = extract_metadata(issue_conllu_path)

    nested: Dict[str, Dict[str, Dict[str, int]]] = {}

    for (pos, word, feats), count in counts.items():
        nested.setdefault(pos, {}).setdefault(word, {})[feats] = count

    return {
        "journal": metadata["journal"],
        "year": metadata["year"],
        "date": metadata["date"],
        "file": str(issue_conllu_path),
        "counts": nested,
        "total_words": total_words,
    }


def preprocess_corpus(
    corpus_path: Path,
    output_path: Path,
    restrict_to_rules: bool = True,
    rules_path: Path | None = None,
    n_jobs: int = -1,
) -> None:
    """Preprocess all files in a corpus in parallel via preprocess.

    Caching this once avoids re-parsing the corpus on every
    downstream change (e.g. a new orthography rule).

    Args:
        corpus_path: Path to the root directory containing the
            corpus files.
        output_path: Path of the JSON file to write (parent
            directory created if missing).
        restrict_to_rules: If True, keep only word forms referenced
            by rules_path. Defaults to True.
        rules_path: Path to the rules CSV used when
            restrict_to_rules is True. Defaults to None, which uses
            config.RULES_FILE.
        n_jobs: Number of parallel jobs passed to joblib.Parallel.
            Defaults to -1 (use all available CPUs).

    Raises:
        FileNotFoundError: If the specified corpus path does not
            exist.
    """

    allowed_forms = (
        _load_allowed_forms(rules_path or RULES_FILE)
        if restrict_to_rules else None
    )

    scanner = FileSystemScanner(corpus_path)
    files = list(scanner.scan())

    tasks = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_preprocess_issue_record)(path, allowed_forms)
        for path in files
    )

    records = list(
        tqdm(tasks, total=len(files), desc="Preprocessing corpus")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _validate_json_cache_shape(
    records: list[dict], output_path: Path
) -> None:
    """Raise ValueError if records use the legacy pre-FEATS shape.

    Args:
        records: Per-issue dicts loaded from a JSON cache file, each
            with a "counts" key.
        output_path: Path the records were loaded from, used to
            build an actionable error message.

    Raises:
        ValueError: If a legacy 2-level "counts" shape is detected.
    """

    for record in records:

        counts = record.get("counts")

        if not counts:
            continue

        word_to_value = next(iter(counts.values()))

        if not word_to_value:
            continue

        value = next(iter(word_to_value.values()))

        if not isinstance(value, dict):
            raise ValueError(
                f"Stale preprocessed corpus cache at {output_path}: "
                "'counts' is nested 2 levels deep "
                "({pos: {word: count}}), but the current format "
                "expects 3 levels ({pos: {word: {feats: count}}}). "
                "Delete this file and re-run preprocess_corpus (or "
                "rerun main.py, which regenerates the cache "
                "automatically when the file is missing)."
            )

        return


def _select_up_to_n_issues_per_date(
    records: list[dict], max_per_date: int = 2
) -> list[dict]:
    """Keep at most `max_per_date` issue records per publication date.

    Records whose date is unknown (date is None) are dropped. Among
    records sharing a date, only the first `max_per_date` encountered
    in `records` order are kept.

    Args:
        records: Per-issue dicts, each with a "date" key (str or
            None).
        max_per_date: Maximum number of records to keep per date.
            Defaults to 2.

    Returns:
        Filtered list of records, in their original relative order,
            with at most `max_per_date` records per date and no
            undated records.
    """

    selected: list[dict] = []
    counts_per_date: dict[str, int] = {}

    for record in records:
        date = record["date"]

        if date is None or counts_per_date.get(date, 0) >= max_per_date:
            continue

        counts_per_date[date] = counts_per_date.get(date, 0) + 1
        selected.append(record)

    return selected


def load_preprocessed_corpus(
    output_path: Path, max_issues_per_date: int = 2
) -> list[dict]:
    """Load a corpus preprocessed by preprocess_corpus.

    Args:
        output_path: JSON file path written by preprocess_corpus.
        max_issues_per_date: Maximum number of issues to keep per
            publication date. Defaults to 2.

    Returns:
        List of per-issue dicts, at most `max_issues_per_date` per
            publication date, no undated records. See
            _preprocess_issue_record for the exact per-dict shape
            (journal, year, date, file, counts, total_words).

    Raises:
        FileNotFoundError: If output_path does not exist.
    """

    if not output_path.exists():
        raise FileNotFoundError(
            f"Preprocessed corpus not found: {output_path}"
        )

    with output_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    _validate_json_cache_shape(records, output_path)
    return _select_up_to_n_issues_per_date(records, max_issues_per_date)
