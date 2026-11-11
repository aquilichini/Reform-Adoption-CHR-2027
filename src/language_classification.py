from collections import Counter
import json
import os
from pathlib import Path
import types
from typing import Dict, Tuple

from conllu import TokenList, parse_incr
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from config import MODEL_CACHE_DIR
from core.filesystem import FileSystemScanner
from core.metadata import extract_metadata

# Process-local cache of loaded fasttext models. key = (model_name, cache_dir).
_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def _sentence_text(sentence: TokenList) -> str:
    """Reconstruct a sentence's surface text from its tokens.

    Args:
        sentence: Parsed CONLLU sentence (list of token dicts).

    Returns:
        Reconstructed sentence text (str).
    """

    words = []

    for token in sentence:
        # ignore multiword / empty nodes
        if not isinstance(token["id"], int):
            continue

        misc = token["misc"]
        space_after = not (misc and misc.get("SpaceAfter") == "No")
        words.append((token["form"], space_after))

    parts = []

    for index, (form, space_after) in enumerate(words):
        parts.append(form)
        if space_after and index != len(words) - 1:
            parts.append(" ")

    return "".join(parts)


def _is_usable_sentence(
    text: str,
    min_chars: int = 3,
    min_alpha_ratio: float = 0.5,
    max_short_word_ratio: float = 0.5,
    short_word_max_len: int = 2,
) -> bool:
    """Heuristic filter to reject OCR-garbled sentence text.

    Args:
        text: Candidate sentence text.
        min_chars: Minimum stripped length to be usable. Defaults to 3.
        min_alpha_ratio: Minimum alphabetic-to-non-space character ratio to be usable. Defaults to 0.5.
        max_short_word_ratio: Maximum allowed fraction of words with length <= short_word_max_len. Defaults to 0.5.
        short_word_max_len: Word length at or below which a word counts as short. Defaults to 2.

    Returns:
        True if the sentence looks usable, False otherwise.
    """

    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return False
    alpha_count = sum(1 for c in non_space if c.isalpha())
    ratio = alpha_count / len(non_space)
    if ratio < min_alpha_ratio:
        return False
    words = stripped.split()
    short_count = sum(1 for w in words if len(w) <= short_word_max_len)
    short_ratio = short_count / len(words)

    return short_ratio <= max_short_word_ratio


def extract_document_text(conllu_path: Path) -> Tuple[str, int, int]:
    """Extract one line of usable text from a CONLLU file.

    Args:
        conllu_path: Path to the CONLLU file.

    Returns:
        Tuple of:
            - Joined document text (str), newline-free.
            - Total number of sentences in the file (int).
            - Number of sentences kept after filtering (int).
    """

    n_total = 0
    kept = []
    with conllu_path.open("r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            n_total += 1
            text = _sentence_text(sentence)
            if _is_usable_sentence(text):
                kept.append(text)
    document_text = " ".join(kept).replace("\n", " ")
    return document_text, n_total, len(kept)


def _numpy2_safe_array(obj, copy=False, **kwargs):
    """Drop-in np.array replacement tolerant of NumPy >= 2.0.

    Args:
        obj: Object to convert to a numpy array.
        copy: If False, allow a copy if one is required instead of
            raising (restores NumPy 1.x copy=False semantics). Any
            other value is passed straight through to np.array.
            Defaults to False.
        **kwargs: Forwarded to np.array/np.asarray (e.g. dtype).

    Returns:
        numpy.ndarray built from obj.
    """

    if copy is False:
        return np.asarray(obj, **kwargs)
    return np.array(obj, copy=copy, **kwargs)


def _get_model(model_name: str, cache_dir: Path | None = None):
    """Lazily load and cache a fasttext LID model.

    Args:
        model_name: Filename of the model on the NbAiLab/nb-nordic-lid HuggingFace repo, e.g. "nb-nordic-lid.ftz".
        cache_dir: Directory hf_hub_download caches the downloaded model file in. Defaults to None, which uses config.MODEL_CACHE_DIR.

    Returns:
        Loaded fasttext model object.
    """

    resolved_cache_dir = cache_dir or MODEL_CACHE_DIR
    key = (model_name, str(resolved_cache_dir))

    if key not in _MODEL_CACHE:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        import fasttext

        # NumPy >= 2.0 incompatibility workaround
        np_proxy = types.SimpleNamespace(**vars(np))
        np_proxy.array = _numpy2_safe_array
        fasttext.FastText.np = np_proxy

        try:
            path = hf_hub_download(
                "NbAiLab/nb-nordic-lid",
                model_name,
                cache_dir=resolved_cache_dir,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            path = hf_hub_download(
                "NbAiLab/nb-nordic-lid",
                model_name,
                cache_dir=resolved_cache_dir,
            )

        _MODEL_CACHE[key] = fasttext.load_model(path)

    return _MODEL_CACHE[key]


def classify_text(
    text: str,
    model,
    threshold: float = 0.25,
) -> Tuple[str, float]:
    """Classify text as bokmål/nynorsk/unknown via a fasttext model.

    Args:
        text: Text to classify.
        model: Loaded fasttext model (e.g. from _get_model).
        threshold: Confidence threshold passed to model.predict.
            Defaults to 0.25.

    Returns:
        Tuple of:
            - Predicted label (str) with the "__label__" prefix
                stripped, or "unknown" if empty/whitespace text or
                nothing clears the threshold.
            - Confidence score (float), 0.0 for "unknown".
    """

    if not text or not text.strip():
        return "unknown", 0.0

    labels, scores = model.predict(text, threshold=threshold)
    if not labels:
        return "unknown", 0.0
    label = labels[0]
    if label.startswith("__label__"):
        label = label[len("__label__"):]
    return label, float(scores[0])


def classify_document(
    conllu_path: Path,
    model,
    threshold: float = 0.25,
) -> dict:
    """Classify a CONLLU file's language variety.

    Args:
        conllu_path: Path to the CONLLU file.
        model: Loaded fasttext model (e.g. from _get_model).
        threshold: Confidence threshold passed to classify_text.
            Defaults to 0.25.

    Returns:
        Dict with keys "label" (str), "confidence" (float),
            "n_sentences_total" (int), and "n_sentences_used"
            (int).
    """

    text, n_total, n_used = extract_document_text(conllu_path)
    label, confidence = classify_text(text, model, threshold)

    return {
        "label": label,
        "confidence": confidence,
        "n_sentences_total": n_total,
        "n_sentences_used": n_used,
    }


def _classify_issue(
    conllu_path: Path,
    model_name: str,
    threshold: float,
    cache_dir: Path | None = None,
) -> dict:
    """Classify one CONLLU file and attach its issue metadata.

    Args:
        conllu_path: Path to the CONLLU file.
        model_name: Model filename passed to _get_model.
        threshold: Confidence threshold passed to
            classify_document.
        cache_dir: Model cache directory passed to _get_model.
            Defaults to None, which uses config.MODEL_CACHE_DIR.

    Returns:
        Dict with keys "journal", "year", "date" (see
            core.metadata.extract_metadata), "file" (str), and
            either the classify_document result fields, or
            "label": "error" plus an "error" message (str) if
            classification failed.
    """

    metadata = extract_metadata(conllu_path)

    try:
        model = _get_model(model_name, cache_dir)
        result = classify_document(conllu_path, model, threshold)
        return {
            **metadata,
            "file": str(conllu_path),
            **result,
        }
    except Exception as exc:
        return {
            **metadata,
            "file": str(conllu_path),
            "label": "error",
            "error": str(exc),
        }


def classify_corpus(
    corpus_path: Path,
    output_path: Path,
    model_name: str = "nb-nordic-lid.ftz",
    threshold: float = 0.25,
    n_jobs: int = -1,
    cache_dir: Path | None = None,
) -> None:
    """Classify all files in a corpus in parallel.

    Args:
        corpus_path: Path to the root directory containing the
            corpus files.
        output_path: Path where the resulting JSON output file will
            be written.
        model_name: Model filename passed to _get_model. Defaults
            to "nb-nordic-lid.ftz".
        threshold: Confidence threshold passed to
            classify_document. Defaults to 0.25.
        n_jobs: Number of parallel jobs passed to joblib.Parallel.
            Defaults to -1 (use all available CPUs).
        cache_dir: Model cache directory passed to _get_model in
            each worker. Defaults to None, which uses
            config.MODEL_CACHE_DIR.

    Raises:
        FileNotFoundError: If the specified corpus path does not
            exist.
    """

    scanner = FileSystemScanner(corpus_path)
    files = list(scanner.scan())

    tasks = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_classify_issue)(
            file, model_name, threshold, cache_dir
        )
        for file in files
    )
    results = list(
        tqdm(tasks, total=len(files), desc="Classifying corpus")
    )
    error_records = [r for r in results if r.get("label") == "error"]
    if error_records:
        print(
            f"Warning: {len(error_records)} of {len(results)} "
            "file(s) failed classification and were labeled "
            f"\"error\". Example ({error_records[0]['file']}): "
            f"{error_records[0]['error']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_classified_corpus(output_path: Path) -> list[dict]:
    """Load a corpus classified by classify_corpus.

    Args:
        output_path: Path to the JSON file written by
            classify_corpus.
    Returns:
        List of per-issue dicts, one per classified CONLLU file
            (journal, year, date, file, label, confidence,
            n_sentences_total, n_sentences_used).

    Raises:
        FileNotFoundError: If output_path does not exist.
    """

    if not output_path.exists():
        raise FileNotFoundError(
            f"Classified corpus not found: {output_path}"
        )

    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_path(path) -> str:
    """Normalize a path string for path-equality comparisons.

    Args:
        path: Path or path-like string.

    Returns:
        Normalized path (str): separators canonicalized and
            redundant "."/".." segments and trailing slashes
            collapsed.
    """

    return os.path.normpath(str(Path(path)))


def attach_labels(
    results: list[dict],
    classified_issues: list[dict],
) -> None:
    """Attach classify_corpus labels onto analyzer results in place.

    Args:
        results: Per-issue analyzer dicts (each with a "file" key),
            mutated in place by setting "label" on every entry.
        classified_issues: Per-issue dicts from
            load_classified_corpus (each with "file" and "label"
            keys).

    Returns:
        None. Mutates each dict in `results` in place.
    """

    label_by_file = {
        _normalize_path(rec["file"]): rec["label"]
        for rec in classified_issues
    }

    matched = 0
    path_matched = 0

    for result in results:
        key = _normalize_path(result["file"])
        if key in label_by_file:
            path_matched += 1
        label = label_by_file.get(key, "unknown")
        result["label"] = label
        if label not in ("unknown", "error"):
            matched += 1

    if classified_issues and results and matched == 0:
        if path_matched == 0:
            print(
                f"WARNING: none of the {len(results)} preprocessed "
                f"issues matched any of the {len(classified_issues)} "
                "classified issues by file path; every issue will be "
                "labeled \"unknown\" and every language-group filter "
                "will end up empty. This usually means the two "
                "caches were built from different corpus roots or "
                "one is stale. try deleting and regenerating both."
            )
        else:
            label_counts = dict(
                Counter(rec["label"] for rec in classified_issues)
            )
            print(
                f"WARNING: {path_matched} of {len(results)} "
                "preprocessed issues matched a classified issue by "
                "file path, but none of the matched labels were "
                "usable (all \"unknown\"/\"error\"); every "
                "language-group filter will end up empty. Label "
                f"distribution across classified issues: "
                f"{label_counts}. This points at classify_corpus "
                "itself, not the join. check why it produced "
                "these labels (e.g. a missing or failing LID model "
                "call)."
            )
