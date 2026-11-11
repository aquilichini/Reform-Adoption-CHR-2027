from __future__ import annotations

from pathlib import Path
from typing import Any

from .filesystem import FileSystemScanner
from .processor import Processor


class Pipeline:
    """
    Main application pipeline.

    Responsibilities:
        - discover files
        - process them
        - return the results
    """

    def __init__(
        self,
        scanner: FileSystemScanner,
        processor: Processor,
    ):
        self.scanner = scanner
        self.processor = processor

    def run(self) -> dict[Path, list[Any]]:
        """
        Execute the complete pipeline.
        """

        files = self.scanner.scan()

        return self.processor.process_all(files)
