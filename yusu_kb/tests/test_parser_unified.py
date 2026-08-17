"""Tests for the unified local parser (all supported formats)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from yusu_kb.knowledge.parser.unified import Parser, is_supported_file_extension
from yusu_kb.knowledge.parser.zip_utils import parse_zip_to_markdown

TXT_CONTENT = "这是第一行。\n这是第二行。"


@pytest.fixture()
def sample_files(tmp_path):
    files = {}

    txt = tmp_path / "sample.txt"
    txt.write_text(TXT_CONTENT, encoding="utf-8")
    files["txt"] = str(txt)

    md = tmp_path / "sample.md"
    md.write_text("# 标题\n\n正文内容。", encoding="utf-8")
    files["md"] = str(md)

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps({"name": "测试", "items": [1, 2]}, ensure_ascii=False), encoding="utf-8")
    files["json"] = str(json_path)

    html = tmp_path / "sample.html"
    html.write_text("<html><body><h1>标题</h1><p>正文段落。</p></body></html>", encoding="utf-8")
    files["html"] = str(html)

    csv = tmp_path / "sample.csv"
    csv.write_text("姓名,年龄\n张三,20\n李四,30\n", encoding="utf-8")
    files["csv"] = str(csv)

    xlsx_path = tmp_path / "sample.xlsx"
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "数据"
    ws.append(["姓名", "年龄"])
    ws.append(["张三", 20])
    ws.append(["李四", 30])
    wb.save(xlsx_path)
    files["xlsx"] = str(xlsx_path)

    pdf = tmp_path / "sample.pdf"
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF text content")
    doc.save(pdf)
    doc.close()
    files["pdf"] = str(pdf)

    docx = tmp_path / "sample.docx"
    import zipfile as docx_zip

    with docx_zip.ZipFile(docx, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr>"
            "<w:r><w:t>文档标题</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>段落一。</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
    files["docx"] = str(docx)

    pptx = tmp_path / "sample.pptx"
    import zipfile as zf

    with zf.ZipFile(pptx, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            "<p:sp><p:txBody><a:p xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>"
            "<a:r><a:t>幻灯片标题</a:t></a:r></a:p></p:txBody></p:sp></p:sld>",
        )
    files["pptx"] = str(pptx)

    return files


class TestExtensionSupport:
    def test_supported_extensions(self):
        for filename in ("a.txt", "b.md", "c.docx", "d.html", "e.htm", "f.json", "g.csv", "h.xlsx", "i.pdf", "j.pptx", "k.zip"):
            assert is_supported_file_extension(filename), filename

    def test_unsupported_extensions(self):
        for filename in ("a.doc", "b.xls", "c.jpg", "d.png", "e.exe"):
            assert not is_supported_file_extension(filename), filename

    def test_case_insensitive(self):
        assert is_supported_file_extension("A.PDF")


@pytest.mark.asyncio
async def test_parse_txt(sample_files):
    content = await Parser.aparse(sample_files["txt"])
    assert "第一行" in content


@pytest.mark.asyncio
async def test_parse_markdown_passthrough(sample_files):
    content = await Parser.aparse(sample_files["md"])
    assert "# 标题" in content


@pytest.mark.asyncio
async def test_parse_json(sample_files):
    content = await Parser.aparse(sample_files["json"])
    assert "测试" in content


@pytest.mark.asyncio
async def test_parse_html(sample_files):
    content = await Parser.aparse(sample_files["html"])
    assert "正文段落" in content


@pytest.mark.asyncio
async def test_parse_csv_markdown_table(sample_files):
    content = await Parser.aparse(sample_files["csv"])
    assert "|" in content
    assert "张三" in content


@pytest.mark.asyncio
async def test_parse_csv_without_tables(sample_files):
    content = await Parser.aparse(sample_files["csv"], {"table_to_markdown": False})
    assert "张三" in content
    assert "|" not in content


@pytest.mark.asyncio
async def test_parse_csv_gbk_encoding(tmp_path):
    path = tmp_path / "gbk.csv"
    path.write_bytes("姓名,城市\n王五,北京\n".encode("gbk"))
    content = await Parser.aparse(str(path))
    assert "王五" in content


@pytest.mark.asyncio
async def test_parse_xlsx_markdown_table(sample_files):
    content = await Parser.aparse(sample_files["xlsx"])
    assert "|" in content
    assert "张三" in content


@pytest.mark.asyncio
async def test_parse_pdf(sample_files):
    content = await Parser.aparse(sample_files["pdf"])
    assert "PDF text content" in content


@pytest.mark.asyncio
async def test_parse_docx(sample_files):
    content = await Parser.aparse(sample_files["docx"])
    assert "段落一" in content


@pytest.mark.asyncio
async def test_parse_pptx(sample_files):
    content = await Parser.aparse(sample_files["pptx"])
    assert "幻灯片标题" in content


class TestZip:
    def test_parse_zip_with_text_and_image(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("doc.md", "# 压缩包文档\n\n正文。")
            archive.writestr("images/pic.png", b"\x89PNG fake")
        saved: dict[str, bytes] = {}

        def save_image(name: str, data: bytes) -> None:
            saved[name] = data

        markdown = parse_zip_to_markdown(buffer.getvalue(), save_image)
        assert "压缩包文档" in markdown
        assert "pic.png" in saved
        assert "images/pic.png" in markdown

    def test_zip_path_traversal_rejected(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../evil.txt", "boom")
            archive.writestr("ok.md", "# OK")
        markdown = parse_zip_to_markdown(buffer.getvalue(), lambda name, data: None)
        assert "OK" in markdown
        assert "boom" not in markdown

    def test_zip_with_no_supported_entries(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("data.bin", b"\x00\x01")
        with pytest.raises(ValueError):
            parse_zip_to_markdown(buffer.getvalue(), lambda name, data: None)

    def test_not_a_zip(self):
        with pytest.raises(ValueError):
            parse_zip_to_markdown(b"this is not a zip", lambda name, data: None)


@pytest.mark.asyncio
async def test_parse_unsupported_extension(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        await Parser.aparse(str(path))


@pytest.mark.asyncio
async def test_parse_missing_file():
    with pytest.raises(ValueError):
        await Parser.aparse("C:/definitely/not/here.pdf")


@pytest.mark.asyncio
async def test_parse_ocr_refused(tmp_path):
    path = tmp_path / "doc.pdf"
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "text")
    doc.save(path)
    doc.close()
    with pytest.raises(ValueError):
        await Parser.aparse(str(path), {"multimodal_ocr": True})


@pytest.mark.asyncio
async def test_aparse_to_result(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# 标题", encoding="utf-8")
    result = await Parser.aparse_to_result(str(path))
    assert result.source_file_path == str(path)
    assert result.markdown_content == "# 标题"