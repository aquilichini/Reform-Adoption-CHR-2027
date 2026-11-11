from __future__ import annotations

import re
from pathlib import Path


def extract_metadata(path: Path) -> dict:
    """Extract journal/year/date metadata from a CONLLU file path.

    Supports two corpus layouts:

    1. Full corpus, e.g.::

           .../klassekampen/1922/19220729_klassekampen.conllu

    2. Test corpus, e.g.::

           vestby_avis_1980_19801127_vestby_avis.conllu

    Args:
        path: Path to the CONLLU file.

    Returns:
        Dict with keys "journal" (str), "year" (int), and "date"
            (str or None).

    Raises:
        ValueError: If neither layout matches the given path.
    """

    # ==========================================================
    # Case 1 : full corpus
    # ==========================================================

    parent = path.parent.name
    if re.fullmatch(r"\d{4}", parent):
        year = int(parent)
        journal = path.parent.parent.name
        filename = path.stem
        match = re.match(r"(\d{8})", filename)
        if match:
            date = match.group(1)
        else:
            date = None
        return {
            "journal": journal,
            "year": year,
            "date": date,
        }

    # ==========================================================
    # Case 2 : test corpus
    # ==========================================================

    stem = path.stem
    match = re.search(r"_(\d{4})_(\d{8})_", stem)
    if match is None:
        raise ValueError(
            f"Cannot extract metadata from filename: {path}"
        )
    year = int(match.group(1))
    date = match.group(2)
    journal = stem[: match.start()]
    return {
        "journal": journal,
        "year": year,
        "date": date,
    }
