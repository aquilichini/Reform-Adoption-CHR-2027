from __future__ import annotations

from pathlib import Path
from typing import Iterable, Callable, Any

from .rule import Rule


class Processor:
    """
    Apply rules to files and execute one or more processors.
    """

    def __init__(
        self,
        rules: Iterable[Rule] | None = None,
        handlers: Iterable[Callable[[Path], Any]] | None = None,
    ):
        self.rules = list(rules or [])
        self.handlers = list(handlers or [])

    def accepts(self, path: Path) -> bool:
        """
        Returns True if the file satisfies every rule.
        """
        return all(rule.match(path) for rule in self.rules)

    def process(self, path: Path) -> list[Any]:
        """
        Apply every handler to one file.
        """
        if not self.accepts(path):
            return []
        results = []
        for handler in self.handlers:
            results.append(handler(path))

        return results

    def process_all(
        self,
        files: Iterable[Path],
    ) -> dict[Path, list[Any]]:
        """
        Process a collection of files.
        """

        output = {}
        for path in files:
            result = self.process(path)
            if result:
                output[path] = result

        return output
