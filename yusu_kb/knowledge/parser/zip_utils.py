"""ZIP archive extraction to markdown.

Ported from YUSU ``yuxi.knowledge.parser.zip_utils`` (demo edition):
file-archive support only (the OLE/docx branch is served by dedicated
parsers and not routed here), images are written through a caller-supplied
callback into the parsed-markdown directory so relative links resolve.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable
from pathlib import PurePosixPath

from yusu_kb.utils.logger import logger

SaveImageCallback = Callable[[str, bytes], None]

_TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tiff", ".tif"}


def _is_safe_relative(relative_path: str) -> bool:
    """Reject absolute paths, drive letters and parent-directory escapes."""
    if not relative_path or relative_path.startswith(("/", "\\")):
        return False
    if re.match(r"^[A-Za-z]:", relative_path):
        return False
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    return ".." not in pure.parts


def _extract_file_archive(archive: zipfile.ZipFile, save_image: SaveImageCallback) -> str:
    """Extract a plain file archive into a single markdown document."""
    sections: list[str] = []
    seen: set[str] = set()

    for info in archive.infolist():
        if info.is_dir():
            continue

        name = info.filename
        if not _is_safe_relative(name):
            logger.warning(f"zip: skipping unsafe entry {name!r}")
            continue

        ext = PurePosixPath(name).suffix.lower()

        if ext in _TEXT_EXTENSIONS:
            if name in seen:
                continue
            seen.add(name)
            try:
                raw = archive.read(info)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                logger.warning(f"zip: failed to read {name}: {exc}")
                continue
            text = _decode_text(raw)
            display_name = PurePosixPath(name).name
            sections.append(f"## {display_name}\n\n{text.strip()}\n")

        elif ext in _IMAGE_EXTENSIONS:
            try:
                raw = archive.read(info)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                logger.warning(f"zip: failed to read image {name}: {exc}")
                continue
            image_filename = PurePosixPath(name).name
            try:
                save_image(image_filename, raw)
            except Exception as exc:  # noqa: BLE001 - one bad image must not abort the archive
                logger.warning(f"zip: failed to save image {name}: {exc}")
                continue
            sections.append(f"![{image_filename}](images/{image_filename})\n")

        else:
            logger.info(f"zip: skipping unsupported entry {name}")

    if not sections:
        raise ValueError("压缩包内没有可提取的文本或图片文件")

    return "\n\n".join(sections)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_zip_to_markdown(zip_bytes: bytes, save_image: SaveImageCallback) -> str:
    """Parse a zip archive (as bytes) into markdown.

    Text/markdown entries are inlined as sections; image entries are handed
    to ``save_image(name, data)`` and referenced as ``images/<name>``.
    """
    if not isinstance(zip_bytes, (bytes, bytearray)):
        raise TypeError("zip_bytes must be bytes")

    if zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            return _extract_file_archive(archive, save_image)

    raise ValueError("不是有效的 ZIP 文件")