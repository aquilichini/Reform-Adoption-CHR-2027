from pathlib import Path
import pandas as pd
from orthography_change.models.rule import OrthographyRule


# ============================================================================
# Required CSV columns
# ============================================================================

REQUIRED_COLUMNS = (
    "rule_id",
    "reform_id",
    "reform_year",
    "lemma",
    "old_form",
    "new_form",
    "language",
)


# ============================================================================
# Validation
# ============================================================================


def validate_rules_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the structure of a rules dataframe.

    The function verifies that every required column is present.
    It raises an exception immediately if the CSV does not conform to the
    expected schema.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


# ============================================================================
# Loading
# ============================================================================


def _normalize_optional_cell(value) -> str | None:
    """Normalize an optional CSV cell to a string or None.

    Missing columns (attribute absent on the row) surface as None
    via getattr's default, and blank/NaN cells (present column, empty
    value) surface as float NaN via pandas; both are normalized to
    None so downstream code has one falsy representation to check.
    """

    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    return value


def load_rules(
    csv_file: str | Path,
) -> list[OrthographyRule]:
    csv_file = Path(csv_file)
    dataframe = pd.read_csv(csv_file)
    validate_rules_dataframe(dataframe)
    rules: list[OrthographyRule] = []
    for row in dataframe.itertuples(index=False):
        comment = _normalize_optional_cell(
            getattr(row, "comment", None)
        )
        pos_filter = _normalize_optional_cell(
            getattr(row, "pos_filter", None)
        )
        feats_filter = _normalize_optional_cell(
            getattr(row, "feats_filter", None)
        )

        rule = OrthographyRule(
            rule_id=row.rule_id,
            reform_id=row.reform_id,
            reform_year=int(row.reform_year),
            lemma=row.lemma,
            old_form=row.old_form,
            new_form=row.new_form,
            language=row.language,
            comment=comment,
            pos_filter=pos_filter,
            feats_filter=feats_filter,
        )

        rules.append(rule)

    return rules


# ============================================================================
# Rule indexing
# ============================================================================


def load_rules_by_old_form(
    csv_file: str | Path,
) -> dict[str, OrthographyRule]:
    """Load rules indexed by their historical spelling.
    """

    rules = load_rules(csv_file)
    index: dict[str, OrthographyRule] = {}
    for rule in rules:
        key = rule.old_form.strip().lower()
        existing = index.get(key)
        if existing is not None and existing.new_form != rule.new_form:
            raise ValueError(
                f"Conflicting new_form for old_form "
                f"{rule.old_form!r}: {existing.new_form!r} vs "
                f"{rule.new_form!r}"
            )

        index.setdefault(key, rule)

    return index


def load_rules_by_new_form(
    csv_file: str | Path,
) -> dict[str, OrthographyRule]:
    """Load rules indexed by their modern spelling.

    This index allows constant-time lookup when scanning a corpus.
    The same new_form may legitimately appear across multiple rules,
    either re-attested across several reforms or because several
    distinct old_form spellings (e.g. inflected variants of one
    lemma) normalize to the same new_form within a single reform.
    Unlike load_rules_by_old_form, no conflict check is performed
    here: a differing old_form for the same new_form is expected
    many-to-one behaviour, not a data error. The first rule seen for
    a given new_form is kept.
    """

    rules = load_rules(csv_file)

    index: dict[str, OrthographyRule] = {}

    for rule in rules:

        index.setdefault(rule.new_form.strip().lower(), rule)

    return index


def load_rules_by_lemma(
    csv_file: str | Path,
) -> dict[str, list[OrthographyRule]]:
    """Group rules by lemma.

    Unlike the previous indices, multiple rules may legitimately share
    the same lemma (for example, successive orthographic reforms).
    """

    rules = load_rules(csv_file)
    index: dict[str, list[OrthographyRule]] = {}
    for rule in rules:
        index.setdefault(rule.lemma, []).append(rule)

    return index


# ============================================================================
# Convenience functions
# ============================================================================


def get_rule_by_old_form(
    old_form: str,
    index: dict[str, OrthographyRule],
) -> OrthographyRule | None:
    """Retrieve a rule from its historical spelling."""
    return index.get(old_form)


def get_rule_by_new_form(
    new_form: str,
    index: dict[str, OrthographyRule],
) -> OrthographyRule | None:
    """Retrieve a rule from its modern spelling."""

    return index.get(new_form)


def get_rules_for_lemma(
    lemma: str,
    index: dict[str, list[OrthographyRule]],
) -> list[OrthographyRule]:
    """Retrieve every rule associated with one lemma."""

    return index.get(lemma, [])
