"""Local filesystem storage — replaces YUSU's MinIO object store.

Layout under the YUSU data directory::

    yusu_data/
    ├── yusu.db
    └── files/
        ├── originals/<file_id>/<filename>   # uploaded raw files
        └── parsed/<file_id>.md              # parsed markdown artifacts

All entries are keyed by server-generated ``file_id``; filenames are
sanitized to prevent path traversal.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ...utils.logger import logger

_PARSE_EXT = ".md"


def sanitize_filename(name: str) -> str:
    """Strip path separators and control characters from a user-supplied filename."""
    name = Path(name).name
    name = re.sub(r"[\\/:\x00-\x1f]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:512] or "unnamed"


class LocalFileStorage:
    """File storage rooted at ``base_dir`` (the ``files/`` directory)."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.originals_dir = self.base_dir / "originals"
        self.parsed_dir = self.base_dir / "parsed"
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    # -- originals -----------------------------------------------------

    def _original_path(self, file_id: str, filename: str) -> Path:
        safe_id = sanitize_filename(file_id)
        return self.originals_dir / safe_id / sanitize_filename(filename)

    def save_file(self, file_id: str, filename: str, data: bytes) -> Path:
        """Persist an uploaded file; returns the stored path."""
        path = self._original_path(file_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.info("saved file %s -> %s", file_id, path)
        return path

    def read_file(self, file_id: str, filename: str) -> bytes | None:
        """Read an uploaded file back, or ``None`` when missing."""
        path = self._original_path(file_id, filename)
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete_file(self, file_id: str, filename: str) -> bool:
        """Delete an uploaded file and its (possibly empty) directory."""
        path = self._original_path(file_id, filename)
        if not path.is_file():
            return False
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True

    def file_exists(self, file_id: str, filename: str) -> bool:
        return self._original_path(file_id, filename).is_file()

    def file_size(self, file_id: str, filename: str) -> int:
        path = self._original_path(file_id, filename)
        return path.stat().st_size if path.is_file() else 0

    # -- parsed markdown -----------------------------------------------

    def _parsed_path(self, file_id: str) -> Path:
        return self.parsed_dir / f"{sanitize_filename(file_id)}{_PARSE_EXT}"

    def save_parsed(self, file_id: str, markdown: str) -> Path:
        """Persist the parsed markdown artifact of a document."""
        path = self._parsed_path(file_id)
        path.write_text(markdown, encoding="utf-8")
        return path

    def read_parsed(self, file_id: str) -> str | None:
        path = self._parsed_path(file_id)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def delete_parsed(self, file_id: str) -> bool:
        path = self._parsed_path(file_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def parsed_exists(self, file_id: str) -> bool:
        return self._parsed_path(file_id).is_file()

    # -- misc ----------------------------------------------------------

    def delete_all(self, file_id: str) -> None:
        """Delete every artifact (originals + parsed) for a file id."""
        parsed = self._parsed_path(file_id)
        if parsed.is_file():
            parsed.unlink()
        originals = self.originals_dir / sanitize_filename(file_id)
        if originals.is_dir():
            for child in originals.iterdir():
                if child.is_file():
                    child.unlink()
            try:
                originals.rmdir()
            except OSError:
                pass

    def originals_under(self, file_id: str) -> list[str]:
        """List stored filenames for a file id."""
        originals = self.originals_dir / sanitize_filename(file_id)
        if not originals.is_dir():
            return []
        return [entry.name for entry in originals.iterdir() if entry.is_file()]

    def total_size(self) -> int:
        """Total bytes stored under this root (excluding parsed artifacts)."""
        total = 0
        for root, _dirs, files in os.walk(self.originals_dir):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
        return total


__all__ = ["LocalFileStorage", "sanitize_filename"]