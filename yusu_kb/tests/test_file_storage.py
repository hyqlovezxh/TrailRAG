"""Tests for the local filesystem storage."""

from __future__ import annotations


def test_sanitize_filename_blocks_traversal():
    from yusu_kb.storage.files.storage import sanitize_filename

    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("a\\b\\c.txt") == "c.txt"
    assert ":" not in sanitize_filename("c:windows")
    assert sanitize_filename("  ") == "unnamed"


def test_save_read_delete_roundtrip(file_storage):
    file_storage.save_file("f1", "doc.md", b"hello world")
    assert file_storage.file_exists("f1", "doc.md")
    assert file_storage.read_file("f1", "doc.md") == b"hello world"
    assert file_storage.file_size("f1", "doc.md") == 11
    assert file_storage.originals_under("f1") == ["doc.md"]

    assert file_storage.delete_file("f1", "doc.md") is True
    assert file_storage.file_exists("f1", "doc.md") is False
    assert file_storage.read_file("f1", "doc.md") is None
    assert file_storage.delete_file("f1", "doc.md") is False


def test_parsed_markdown_roundtrip(file_storage):
    file_storage.save_parsed("f2", "# Title\n\n正文")
    assert file_storage.parsed_exists("f2")
    assert file_storage.read_parsed("f2") == "# Title\n\n正文"
    assert file_storage.delete_parsed("f2") is True
    assert file_storage.read_parsed("f2") is None


def test_delete_all(file_storage):
    file_storage.save_file("f3", "a.md", b"data")
    file_storage.save_parsed("f3", "parsed")
    file_storage.delete_all("f3")
    assert not file_storage.file_exists("f3", "a.md")
    assert not file_storage.parsed_exists("f3")
    assert file_storage.originals_under("f3") == []


def test_total_size(file_storage):
    file_storage.save_file("f4", "a.md", b"12345")
    file_storage.save_file("f4", "b.md", b"123")
    assert file_storage.total_size() == 8
