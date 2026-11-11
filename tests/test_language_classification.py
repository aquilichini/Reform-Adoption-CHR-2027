import json
from pathlib import Path

import numpy as np
import pytest
from conllu import parse

import language_classification
from language_classification import (
    _is_usable_sentence,
    _numpy2_safe_array,
    _sentence_text,
    attach_labels,
    classify_corpus,
    classify_text,
    extract_document_text,
    load_classified_corpus,
)


# ============================================================================
# Helpers
# ============================================================================


CONLLU_TEXT = """\
# sent_id = 1
# text = Katten sover godt.
1\tKatten\tkatt\tNOUN\t_\t_\t0\troot\t_\t_
2\tsover\tsove\tVERB\t_\t_\t1\tconj\t_\t_
3\tgodt\tgodt\tADV\t_\t_\t1\tadvmod\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_

# sent_id = 2
# text = 12 34 56 ...
1\t12\t12\tNUM\t_\t_\t0\troot\t_\t_
2\t34\t34\tNUM\t_\t_\t0\troot\t_\t_
3\t56\t56\tNUM\t_\t_\t0\troot\t_\t_
4\t...\t...\tPUNCT\t_\t_\t0\troot\t_\t_
"""

MALFORMED_CONLLU_TEXT = "this is not valid conllu\t\t\x00\x01\n"

# Real OCR-garbled corpus content: parses fine as CONLLU but its
# reconstructed sentence text is a string of 1-2 letter pseudo-words
# and should be dropped by _is_usable_sentence.
GIBBERISH_CONLLU_TEXT = """\
# text = e k s. u t eea e iea ; e s e. e e s :
1\te k\te k\tX\t_\t_\t0\troot\t_\t_
2\ts. u t\ts. u t\tX\t_\t_\t3\tpunct\t_\t_
3\teea\teea\tX\t_\t_\t1\tflat:foreign\t_\t_
4\te\tvære\tX\t_\t_\t3\tflat:foreign\t_\t_
5\tiea\tiea\tINTJ\t_\t_\t3\tflat:foreign\t_\t_
6\t;\t$;\tPUNCT\t_\t_\t5\tpunct\t_\t_
7\te s e. e e s\te s e. e e s\tNOUN\t_\tAbbr=Yes\t5\tflat:foreign\t_\t_
8\t:\t$:\tPUNCT\t_\t_\t1\tpunct\t_\t_
"""


def write_conllu(path: Path, name: str, text: str = CONLLU_TEXT) -> Path:
    """Write a small CONLLU fixture file.

    Args:
        path: Directory to write the file into.
        name: Filename (without directory) for the fixture.
        text: Contents to write. Defaults to CONLLU_TEXT.

    Returns:
        Path to the written file.
    """

    file_path = path / name
    file_path.write_text(text, encoding="utf-8")

    return file_path


class _FakeModel:
    """Canned fasttext-like model for tests (no network calls)."""

    def __init__(self, label="__label__nob", score=0.95):
        self.label = label
        self.score = score

    def predict(self, text, threshold=0.25):
        if self.score < threshold:
            return (), np.array([])
        return (self.label,), np.array([self.score])


# ============================================================================
# _sentence_text / extract_document_text
# ============================================================================


def test_sentence_text_honors_space_after_no():

    sentences = parse(CONLLU_TEXT)

    text = _sentence_text(sentences[0])

    assert text == "Katten sover godt."


def test_extract_document_text_drops_garbled_sentence(tmp_path):

    file_path = write_conllu(tmp_path, "issue.conllu")

    text, n_total, n_used = extract_document_text(file_path)

    assert n_total == 2
    assert n_used == 1
    assert text == "Katten sover godt."
    assert "\n" not in text


def test_extract_document_text_drops_gibberish_sentence(tmp_path):

    file_path = write_conllu(
        tmp_path, "issue.conllu", text=GIBBERISH_CONLLU_TEXT
    )

    text, n_total, n_used = extract_document_text(file_path)

    assert n_total == 1
    assert n_used == 0
    assert text == ""


# ============================================================================
# _is_usable_sentence
# ============================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", False),
        ("   ", False),
        ("Hi", False),
        ("12 34 56 ...", False),
        ("Katten sover godt.", True),
        # Real OCR-garbled corpus text: 1-2 letter pseudo-words.
        ("e k s. u t eea e iea ; e s e. e e s :", False),
    ],
)
def test_is_usable_sentence(text, expected):

    assert _is_usable_sentence(text) is expected


# ============================================================================
# _numpy2_safe_array
# ============================================================================


def test_numpy2_safe_array_copy_false_does_not_raise():

    # Regression check: under real numpy>=2, np.array(list,
    # copy=False) raises ValueError. This is the exact call fasttext
    # 0.9.2's predict() makes.
    result = _numpy2_safe_array([0.1, 0.2, 0.3], copy=False)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


def test_numpy2_safe_array_copy_true_behaves_like_np_array():

    result = _numpy2_safe_array([0.1, 0.2, 0.3], copy=True)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


# ============================================================================
# classify_text
# ============================================================================


def test_classify_text_confident_prediction():

    model = _FakeModel(label="__label__nob", score=0.95)

    label, confidence = classify_text(
        "Debatt er bra og sunt.", model, threshold=0.25
    )

    assert label == "nob"
    assert confidence == pytest.approx(0.95)


def test_classify_text_below_threshold_is_unknown():

    model = _FakeModel(label="__label__nno", score=0.1)

    label, confidence = classify_text(
        "Ukjent tekst.", model, threshold=0.25
    )

    assert label == "unknown"
    assert confidence == 0.0


def test_classify_text_empty_text_is_unknown():

    model = _FakeModel(label="__label__nob", score=0.95)

    label, confidence = classify_text("   ", model, threshold=0.25)

    assert label == "unknown"
    assert confidence == 0.0


# ============================================================================
# classify_corpus
# ============================================================================


def test_classify_corpus_writes_expected_json(tmp_path, monkeypatch):

    monkeypatch.setattr(
        language_classification,
        "_get_model",
        lambda name, cache_dir=None: _FakeModel(
            label="__label__nob", score=0.95
        ),
    )

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "19801127_vestby_avis.conllu")

    output_path = tmp_path / "output" / "classified.json"

    classify_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        n_jobs=1,
    )

    assert output_path.exists()

    results = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(results) == 1

    issue = results[0]

    assert issue["journal"] == "vestby_avis"
    assert issue["year"] == 1980
    assert issue["date"] == "19801127"
    assert issue["label"] == "nob"
    assert issue["confidence"] == pytest.approx(0.95)
    assert issue["n_sentences_total"] == 2
    assert issue["n_sentences_used"] == 1


def test_classify_corpus_handles_malformed_file(tmp_path, monkeypatch):

    monkeypatch.setattr(
        language_classification,
        "_get_model",
        lambda name, cache_dir=None: _FakeModel(
            label="__label__nob", score=0.95
        ),
    )

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(
        corpus_dir,
        "19801127_vestby_avis.conllu",
        text=MALFORMED_CONLLU_TEXT,
    )

    output_path = tmp_path / "output" / "classified.json"

    classify_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        n_jobs=1,
    )

    results = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(results) == 1

    issue = results[0]

    assert issue["label"] == "error"
    assert "error" in issue
    assert issue["journal"] == "vestby_avis"


def test_classify_corpus_prints_warning_on_errors(
    tmp_path, monkeypatch, capsys,
):

    monkeypatch.setattr(
        language_classification,
        "_get_model",
        lambda name, cache_dir=None: _FakeModel(
            label="__label__nob", score=0.95
        ),
    )

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(
        corpus_dir,
        "19801127_vestby_avis.conllu",
        text=MALFORMED_CONLLU_TEXT,
    )

    output_path = tmp_path / "output" / "classified.json"

    classify_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        n_jobs=1,
    )

    out = capsys.readouterr().out

    assert "Warning" in out
    assert "1 of 1" in out
    assert "error" in out


# ============================================================================
# load_classified_corpus
# ============================================================================


def test_load_classified_corpus_round_trip(tmp_path, monkeypatch):

    monkeypatch.setattr(
        language_classification,
        "_get_model",
        lambda name, cache_dir=None: _FakeModel(
            label="__label__nno", score=0.8
        ),
    )

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "19801127_vestby_avis.conllu")

    output_path = tmp_path / "output" / "classified.json"

    classify_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        n_jobs=1,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))

    loaded = load_classified_corpus(output_path)

    assert loaded == written

    assert len(loaded) == 1

    issue = loaded[0]

    assert issue["journal"] == "vestby_avis"
    assert issue["label"] == "nno"


def test_load_classified_corpus_missing_file(tmp_path):

    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):

        load_classified_corpus(missing_path)


# ============================================================================
# _get_model on-disk caching
# ============================================================================


def test_get_model_tries_local_files_only_first(tmp_path, monkeypatch):

    pytest.importorskip("huggingface_hub")
    pytest.importorskip("fasttext")

    calls = []

    def fake_hf_hub_download(
        repo_id, filename, cache_dir=None, local_files_only=False,
        **kwargs,
    ):
        calls.append(local_files_only)
        return str(tmp_path / "fake_model.ftz")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download
    )
    monkeypatch.setattr(
        "fasttext.load_model", lambda path: _FakeModel()
    )
    language_classification._MODEL_CACHE.clear()

    model = language_classification._get_model(
        "nb-nordic-lid.ftz", cache_dir=tmp_path
    )

    assert isinstance(model, _FakeModel)
    assert calls == [True]


def test_get_model_falls_back_to_download_on_cache_miss(
    tmp_path, monkeypatch,
):

    pytest.importorskip("huggingface_hub")
    pytest.importorskip("fasttext")

    from huggingface_hub.utils import LocalEntryNotFoundError

    calls = []

    def fake_hf_hub_download(
        repo_id, filename, cache_dir=None, local_files_only=False,
        **kwargs,
    ):
        calls.append(local_files_only)
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path / "fake_model.ftz")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download
    )
    monkeypatch.setattr(
        "fasttext.load_model", lambda path: _FakeModel()
    )
    language_classification._MODEL_CACHE.clear()

    model = language_classification._get_model(
        "nb-nordic-lid.ftz", cache_dir=tmp_path
    )

    assert isinstance(model, _FakeModel)
    assert calls == [True, False]


def test_get_model_patches_fasttext_np_array(tmp_path, monkeypatch):

    pytest.importorskip("huggingface_hub")
    pytest.importorskip("fasttext")

    import fasttext

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: str(tmp_path / "fake_model.ftz"),
    )
    monkeypatch.setattr(
        "fasttext.load_model", lambda path: _FakeModel()
    )
    language_classification._MODEL_CACHE.clear()

    language_classification._get_model(
        "nb-nordic-lid.ftz", cache_dir=tmp_path
    )

    assert fasttext.FastText.np.array is _numpy2_safe_array


# ============================================================================
# Importable without fasttext / huggingface_hub installed
# ============================================================================


def test_module_usable_without_optional_lid_dependencies():

    # No monkeypatching, no model loaded: proves the module imports
    # and its non-model helpers work without fasttext/huggingface_hub
    # installed.
    assert _is_usable_sentence("Katten sover godt.") is True
    assert _is_usable_sentence("") is False


# ============================================================================
# attach_labels
# ============================================================================


def test_attach_labels_matches_exact_path():

    results = [{"file": Path("a/b/x.conllu")}]
    classified_issues = [{"file": "a/b/x.conllu", "label": "nob"}]

    attach_labels(results, classified_issues)

    assert results[0]["label"] == "nob"


def test_attach_labels_matches_normalized_path():

    results = [{"file": Path("a/b/x.conllu")}]
    classified_issues = [{"file": "a/b/../b/x.conllu", "label": "nno"}]

    attach_labels(results, classified_issues)

    assert results[0]["label"] == "nno"


def test_attach_labels_defaults_to_unknown_when_unmatched():

    results = [{"file": Path("a/b/x.conllu")}]
    classified_issues = [{"file": "a/b/other.conllu", "label": "nob"}]

    attach_labels(results, classified_issues)

    assert results[0]["label"] == "unknown"


def test_attach_labels_warns_on_total_join_failure(capsys):

    results = [{"file": Path("a/b/x.conllu")}]
    classified_issues = [{"file": "a/b/other.conllu", "label": "nob"}]

    attach_labels(results, classified_issues)

    out = capsys.readouterr().out

    assert out != ""
    assert "unknown" in out or "match" in out


def test_attach_labels_no_warning_when_classified_issues_empty(capsys):

    results = [{"file": Path("a/b/x.conllu")}]

    attach_labels(results, [])

    assert results[0]["label"] == "unknown"
    assert capsys.readouterr().out == ""


def test_attach_labels_warns_on_classification_failure(capsys):

    results = [
        {"file": Path("a/b/x.conllu")},
        {"file": Path("a/b/y.conllu")},
    ]
    classified_issues = [
        {"file": "a/b/x.conllu", "label": "error"},
        {"file": "a/b/y.conllu", "label": "error"},
    ]

    attach_labels(results, classified_issues)

    out = capsys.readouterr().out

    assert results[0]["label"] == "error"
    assert results[1]["label"] == "error"
    assert out != ""
    assert "error" in out
    assert "different corpus roots" not in out


def test_attach_labels_no_warning_when_some_match(capsys):

    results = [
        {"file": Path("a/b/x.conllu")},
        {"file": Path("a/b/y.conllu")},
    ]
    classified_issues = [{"file": "a/b/x.conllu", "label": "nob"}]

    attach_labels(results, classified_issues)

    assert results[0]["label"] == "nob"
    assert results[1]["label"] == "unknown"
    assert capsys.readouterr().out == ""
