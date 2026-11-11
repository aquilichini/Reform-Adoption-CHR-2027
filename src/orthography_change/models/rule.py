from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class OrthographyRule:
    """
    Representation of one orthographic change.

    Parameters
        rule_id : str
            Unique identifier of the rule.
        reform_id : str
            Identifier of the orthographic reform.
        reform_year : int
            Official publication year of the reform.
        lemma : str
            Canonical lemma concerned by the reform.
        old_form : str
            Historical spelling.
        new_form : str
            New spelling introduced by the reform.
        language : str
            Written standard the reform belongs to: "nb" (bokmål) or
            "nn" (nynorsk).
        comment : str | None, default=None
            Optional comment associated with the rule.
        pos_filter : str | None, default=None
            Optional UPOS tag required for a token to count toward this
            rule (e.g. "VERB"). Unset means any UPOS matches.
        feats_filter : str | None, default=None
            Optional pipe-separated "Key=Value" morphological features
            required for a token to count toward this rule (e.g.
            "VerbForm=Part"). Unset means any FEATS matches.
    """

    rule_id: str
    reform_id: str
    reform_year: int
    lemma: str
    old_form: str
    new_form: str
    language: str
    comment: str | None = None
    pos_filter: str | None = None
    feats_filter: str | None = None

    @property
    def forms(self) -> tuple[str, str]:
        """
        Return both spellings.

        Returns
        -------
        tuple[str, str]
            Tuple containing (old_form, new_form).
        """

        return self.old_form, self.new_form

    @property
    def label(self) -> str:
        """
        Return a human-readable label describing the rule.

        Returns
        -------
        str
            Label formatted as "old_form --_ new_form".
        """

        return f"{self.old_form} --_ {self.new_form}"

    @property
    def filename(self) -> str:
        """
        Return a filesystem-safe filename stem.

        This value is intended to be used when exporting CSV files,
        figures or other artefacts generated for this rule.

        Returns
        -------
        str
        """

        return f"{self.old_form}_to_{self.new_form}"

    def matches(self, form: str) -> bool:
        """
        Determine whether a token matches this rule.

        Parameters
        ----------
        form : str
            Token form.

        Returns
        -------
        bool
            True if the token corresponds to either the historical
            or the modern spelling.
        """

        return form == self.old_form or form == self.new_form

    def is_old(self, form: str) -> bool:
        """
        Determine whether a token corresponds to the historical spelling.

        Parameters
        ----------
        form : str
            Token form.

        Returns
        -------
        bool
        """

        return form == self.old_form

    def is_new(self, form: str) -> bool:
        """
        Determine whether a token corresponds to the modern spelling.

        Parameters
        ----------
        form : str
            Token form.

        Returns
        -------
        bool
        """

        return form == self.new_form

    def matches_morphology(self, upos: str, feats: str) -> bool:
        """
        Determine whether a token's POS/morphology satisfy this rule.

        Parameters
        ----------
        upos : str
            Token's universal part-of-speech tag.

        feats : str
            Token's canonicalized morphological features, formatted
            as pipe-separated "Key=Value" pairs (see
            preprocessing._feats_to_str).

        Returns
        -------
        bool
            True if pos_filter is unset or equals upos, and every
            "Key=Value" pair in feats_filter is present in feats.
            True for both filters unset (default, unrestricted).
        """

        if self.pos_filter is not None and self.pos_filter != upos:
            return False

        if self.feats_filter is not None:

            token_feats = set(feats.split("|")) if feats else set()
            required_feats = set(self.feats_filter.split("|"))

            if not required_feats.issubset(token_feats):
                return False

        return True

    def adoption_label(self) -> str:
        """
        Return a descriptive label suitable for plots and reports.

        Returns
        -------
        str
        """

        return (
            f"{self.old_form} → {self.new_form} "
            f"({self.reform_id}, {self.reform_year})"
        )
