from __future__ import annotations

from pathlib import Path
from typing import Iterator


class FileSystemScanner:
    """
    Scanne un répertoire et retourne tous les fichiers .conllu.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def scan(self) -> Iterator[Path]:
        """
        Parcourt récursivement le dossier root
        et yield tous les fichiers .conllu.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")

        for path in self.root.rglob("*.conllu"):
            if path.is_file():
                yield path
