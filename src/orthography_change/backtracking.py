from dataclasses import dataclass
from orthography_change.models.rule import OrthographyRule


@dataclass(slots=True, frozen=True)
class BacktrackingEvent:
    """One reform's pull on a backtracking pair.

    Attributes:
        reform_id: Identifier of the reform.
        reform_year: Official publication year of the reform.
        direction: "up" if the reform pushes toward the group's
            anchor new_form, "down" if it pulls back toward the
            anchor old_form.
    """

    reform_id: str
    reform_year: int
    direction: str


@dataclass(slots=True, frozen=True)
class BacktrackingGroup:
    """A form pair pulled in opposite directions by 2+ reforms.

    Attributes:
        old_form: Anchor spelling A (earliest reform's old_form).
        new_form: Anchor spelling B (earliest reform's new_form).
        events: Reforms affecting the pair, sorted by reform_year.
        language: Written standard shared by every rule in the
            pair ("nb" or "nn"); reform_id values are
            language-homogeneous, so a form pair's rules always
            share one language.
    """

    old_form: str
    new_form: str
    events: tuple[BacktrackingEvent, ...]
    language: str

    @property
    def label(self) -> str:
        """Return "old _--_ new", parallel to OrthographyRule.label.

        Returns:
            str: Human-readable label for the pair.
        """

        return f"{self.old_form} _--_ {self.new_form}"


def group_rules_by_form_pair(
    rules: list[OrthographyRule],
) -> dict[frozenset[str], list[OrthographyRule]]:
    """Group rules whose {old_form, new_form} set is identical.

    Args:
        rules: Rules to group.

    Returns:
        dict[frozenset[str], list[OrthographyRule]]: Rules keyed by
            their unordered form pair, in order of first appearance.
    """

    groups: dict[frozenset[str], list[OrthographyRule]] = {}
    for rule in rules:
        key = frozenset({rule.old_form, rule.new_form})
        groups.setdefault(key, []).append(rule)
    return groups


def find_backtracking_groups(
    rules: list[OrthographyRule],
) -> list[BacktrackingGroup]:
    """Find form pairs whose direction reverses across reforms.

    A group qualifies when its rules include more than one distinct
    old_form value, which naturally excludes same-direction
    re-attestation and isolated single rules (both have exactly 1
    distinct old_form / group size 1). The anchor is the
    earliest-reform_year rule's (old_form, new_form), tie-broken by
    reform_id. Handles chains of any length that keep alternating on
    the same unordered pair.

    Args:
        rules: Rules to scan.

    Returns:
        list[BacktrackingGroup]: Groups in order of first appearance
            in `rules`.

    Raises:
        ValueError: If a form pair's rules do not all share the
            same language.
    """

    groups: list[BacktrackingGroup] = []
    for pair_rules in group_rules_by_form_pair(rules).values():
        if len({r.old_form for r in pair_rules}) <= 1:
            continue
        ordered = sorted(
            pair_rules,
            key=lambda r: (r.reform_year, r.reform_id),
        )
        anchor = ordered[0]
        languages = {r.language for r in ordered}
        if len(languages) > 1:
            raise ValueError(
                f"Conflicting language for form pair "
                f"{{{anchor.old_form!r}, {anchor.new_form!r}}}: "
                f"{sorted(languages)!r}"
            )
        events = tuple(
            BacktrackingEvent(
                reform_id=r.reform_id,
                reform_year=r.reform_year,
                direction=(
                    "up" if r.new_form == anchor.new_form else "down"
                ),
            )
            for r in ordered
        )
        groups.append(
            BacktrackingGroup(
                old_form=anchor.old_form,
                new_form=anchor.new_form,
                events=events,
                language=anchor.language,
            )
        )
    return groups


def backtracking_pairs(
    groups: list[BacktrackingGroup],
) -> set[frozenset[str]]:
    """Return the form pairs covered by backtracking groups.

    Args:
        groups: Backtracking groups.

    Returns:
        set[frozenset[str]]: {old_form, new_form} sets, used by
            main.py to skip per-direction plots already covered by
            a combined backtracking plot.
    """

    return {frozenset({g.old_form, g.new_form}) for g in groups}
