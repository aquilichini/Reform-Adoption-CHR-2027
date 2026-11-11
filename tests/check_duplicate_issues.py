"""Standalone diagnostic: look for duplicated issues in the raw
preprocessed corpus cache (before the one-issue-per-date filter in
preprocessing.load_preprocessed_corpus is applied).

Not a pytest test (see tests/metadata.py for the same convention) -
run directly:

    uv run python tests/check_duplicate_issues.py [path/to/cache.json]

Defaults to config.PREPROCESSED_CACHE_JSON when no path is given.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import PREPROCESSED_CACHE_JSON  # noqa: E402


def load_raw_records(path: Path) -> list[dict]:
    """Load a preprocessed corpus cache with no filtering applied.

    Deliberately bypasses preprocessing.load_preprocessed_corpus (it
    now caps at one issue per date), so duplication in the raw cache
    is still visible here.

    Args:
        path: Path to the preprocess_corpus JSON cache.

    Returns:
        List of per-issue dicts, exactly as stored on disk.
    """

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_duplicate_files(records: list[dict]) -> dict[str, int]:
    """Find "file" paths that appear more than once in records.

    Args:
        records: Per-issue dicts, each with a "file" key.

    Returns:
        Dict mapping duplicated file path to its occurrence count
            (only entries with count > 1).
    """

    counts = Counter(record["file"] for record in records)
    return {path: n for path, n in counts.items() if n > 1}


def find_same_journal_date_duplicates(
    records: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group records sharing both journal and date, keep groups > 1.

    Multiple different journals publishing on the same date is
    normal and excluded here; this only flags the same journal
    appearing more than once on the same date, which points at
    genuine duplication rather than expected multi-newspaper
    coverage.

    Args:
        records: Per-issue dicts, each with "journal" and "date"
            keys. Records with date=None are skipped (nothing to
            group them by).

    Returns:
        Dict mapping (journal, date) to its list of records, for
            every (journal, date) pair with more than one record.
    """

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in records:
        if record["date"] is None:
            continue
        groups[(record["journal"], record["date"])].append(record)

    return {key: recs for key, recs in groups.items() if len(recs) > 1}


def duplicates_are_identical(records: list[dict]) -> bool:
    """Check whether a group of duplicate records share content.

    Args:
        records: Records for one (journal, date) group (as returned
            by find_same_journal_date_duplicates).

    Returns:
        True if every record has the same "counts" and
            "total_words" as the first one (byte-for-byte
            duplication of content), False otherwise.
    """

    first = records[0]
    return all(
        r["counts"] == first["counts"]
        and r["total_words"] == first["total_words"]
        for r in records[1:]
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PREPROCESSED_CACHE_JSON

    print(f"Loading raw cache: {path}")
    records = load_raw_records(path)
    print(f"Total records: {len(records)}")

    dup_files = find_duplicate_files(records)
    print(f"\nExact duplicate file paths: {len(dup_files)}")
    for file_path, n in sorted(dup_files.items())[:10]:
        print(f"  {n}x  {file_path}")

    dup_groups = find_same_journal_date_duplicates(records)
    extra = sum(len(recs) - 1 for recs in dup_groups.values())
    print(
        f"\nSame journal + same date duplicate groups: "
        f"{len(dup_groups)} ({extra} extra records beyond one "
        f"per group)"
    )

    identical = sum(
        1 for recs in dup_groups.values() if duplicates_are_identical(recs)
    )
    print(
        f"  Of which byte-identical (same counts/total_words): "
        f"{identical}"
    )
    print(f"  Differing content within the group: "
          f"{len(dup_groups) - identical}")

    for (journal, date), recs in sorted(dup_groups.items())[:10]:
        same = duplicates_are_identical(recs)
        tag = "identical" if same else "differs"
        print(f"  {journal} @ {date}: {len(recs)} records ({tag})")


if __name__ == "__main__":
    main()
