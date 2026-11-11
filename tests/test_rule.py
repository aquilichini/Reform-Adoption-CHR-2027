from orthography_change.models.rule import OrthographyRule


def test_forms():

    rule = OrthographyRule(
        rule_id="R1",
        reform_id="DK1948",
        reform_year=1948,
        lemma="å",
        old_form="aa",
        new_form="å",
        language="nb",
    )

    assert rule.forms == ("aa", "å")


def test_matches():

    rule = OrthographyRule(
        rule_id="R1",
        reform_id="DK1948",
        reform_year=1948,
        lemma="å",
        old_form="aa",
        new_form="å",
        language="nb",
    )

    assert rule.matches("aa")
    assert rule.matches("å")
    assert not rule.matches("oe")


def test_old():

    rule = OrthographyRule(
        rule_id="R1",
        reform_id="DK1948",
        reform_year=1948,
        lemma="å",
        old_form="aa",
        new_form="å",
        language="nb",
    )

    assert rule.is_old("aa")
    assert not rule.is_old("å")


def test_new():

    rule = OrthographyRule(
        rule_id="R1",
        reform_id="DK1948",
        reform_year=1948,
        lemma="å",
        old_form="aa",
        new_form="å",
        language="nb",
    )

    assert rule.is_new("å")
    assert not rule.is_new("aa")
