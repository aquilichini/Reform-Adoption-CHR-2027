from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class Rule(ABC):
    """
    Base class for every file selection rule.
    """

    @abstractmethod
    def match(self, path: Path) -> bool:
        """
        Returns True if the rule accepts the file.
        """
        raise NotImplementedError


class ExtensionRule(Rule):
    """
    Accept only specific extensions.
    """

    def __init__(self, extensions: Iterable[str]):
        self.extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        }

    def match(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions


class NameContainsRule(Rule):
    """
    Accept files whose filename contains a given substring.
    """

    def __init__(self, text: str):
        self.text = text.lower()

    def match(self, path: Path) -> bool:
        return self.text in path.name.lower()


class MinSizeRule(Rule):
    """
    Accept files larger than a minimum size.
    """

    def __init__(self, min_bytes: int):
        self.min_bytes = min_bytes

    def match(self, path: Path) -> bool:
        try:
            return path.stat().st_size >= self.min_bytes
        except OSError:
            return False


class MaxSizeRule(Rule):
    """
    Accept files smaller than a maximum size.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes

    def match(self, path: Path) -> bool:
        try:
            return path.stat().st_size <= self.max_bytes
        except OSError:
            return False


class CompositeRule(Rule):
    """
    Logical AND of several rules.
    """

    def __init__(self, *rules: Rule):
        self.rules = list(rules)

    def match(self, path: Path) -> bool:
        return all(rule.match(path) for rule in self.rules)


class AnyRule(Rule):
    """
    Logical OR of several rules.
    """

    def __init__(self, *rules: Rule):
        self.rules = list(rules)

    def match(self, path: Path) -> bool:
        return any(rule.match(path) for rule in self.rules)


class NotRule(Rule):
    """
    Logical NOT.
    """

    def __init__(self, rule: Rule):
        self.rule = rule

    def match(self, path: Path) -> bool:
        return not self.rule.match(path)
