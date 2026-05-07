import zipfile
from types import SimpleNamespace

from app.crawlers.docling_collector import (
    DoclingCollectorConfig,
    _clean_hwp_text,
    _extract_title,
    collect_documents_with_docling,
)


class _FakeConverter:
    def convert(self, source, headers=None):
        return SimpleNamespace(
            document=SimpleNamespace(export_to_markdown=lambda: "# Example\ncontent")
        )


class _FailingConverter:
    def convert(self, source, headers=None):
        raise RuntimeError("conversion failed")


class _CountingConverter:
    calls = 0

    def convert(self, source, headers=None):
        self.calls += 1
        return SimpleNamespace(
            document=SimpleNamespace(export_to_markdown=lambda: "# Fallback\ncontent")
        )


def test_collect_documents_with_docling_skips_images_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: _FakeConverter(),
    )

    documents = collect_documents_with_docling(
        ["https://example.com/file.png"],
        config=DoclingCollectorConfig(skip_images=True),
    )

    assert documents == []


def test_collect_documents_with_docling_keeps_non_image_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: _FakeConverter(),
    )

    documents = collect_documents_with_docling(
        ["https://example.com/file.pdf"],
        config=DoclingCollectorConfig(skip_images=True),
    )

    assert len(documents) == 1
    assert documents[0].source_type == "pdf"


def test_collect_documents_with_docling_extracts_hwpx_text(tmp_path) -> None:
    hwpx_path = tmp_path / "sample.hwpx"
    with zipfile.ZipFile(hwpx_path, "w") as archive:
        archive.writestr(
            "Contents/section0.xml",
            "<root><p>입체조형학과</p><p>실험실습비 사용내역</p></root>",
        )

    documents = collect_documents_with_docling([str(hwpx_path)])

    assert len(documents) == 1
    assert documents[0].source_type == "hwpx"
    assert "입체조형학과" in documents[0].content
    assert "실험실습비 사용내역" in documents[0].content

def test_clean_hwp_text_removes_binary_noise_and_keeps_korean() -> None:
    cleaned = _clean_hwp_text(
        "Bƀ ࠄ 铒먉 捤獥 汤捯\n"
        "2026-1 성적향상장학금 신청 안내\n"
        "신청기간: 2026.03.09.(월)~20.(금)\n"
        "\x00\x01 KUTIS 로그인 → 신청마당"
    )

    assert "Bƀ" not in cleaned
    assert "铒먉" not in cleaned
    assert "2026-1 성적향상장학금 신청 안내" in cleaned
    assert "신청기간: 2026.03.09.(월)~20.(금)" in cleaned
    assert "KUTIS 로그인 → 신청마당" in cleaned


def test_clean_hwp_text_removes_image_metadata_and_cjk_noise() -> None:
    cleaned = _clean_hwp_text(
        "漠杳 悵 그림입니다.\n"
        "원본 그림의 이름: 투명배경마크.png\n"
        "원본 그림의 크기: 가로 1336pixel, 세로 917pixel\n"
        "프로그램 이름 : Adobe ImageReady\n"
        "B 먉 밼 쀈 d d U < 楣⑰ 蝨 贀 솚 2026-1 성적향상장학금 신청 안내\n"
        "신청 전 확인사항"
    )

    assert "그림입니다" not in cleaned
    assert "원본 그림" not in cleaned
    assert "Adobe ImageReady" not in cleaned
    assert "漠杳" not in cleaned
    assert "B 먉" not in cleaned
    assert "밼 쀈" not in cleaned
    assert "솚 2026-1" not in cleaned
    assert "2026-1 성적향상장학금 신청 안내" in cleaned
    assert "신청 전 확인사항" in cleaned


def test_extract_title_skips_garbled_hwp_lines() -> None:
    title = _extract_title(
        content="Bƀ ࠄ 铒먉 捤獥 汤捯\n2026-1 성적향상장학금 신청 안내\n본문",
        source="https://example.com/downloadBbsFile.do?atchmnflNo=1",
        source_type="hwp",
        headers={},
        timeout_seconds=1,
    )

    assert title == "2026-1 성적향상장학금 신청 안내"


def test_collect_documents_with_docling_extracts_supported_zip_members(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: _FakeConverter(),
    )
    zip_path = tmp_path / "attachments.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("notice.pdf", b"%PDF")
        archive.writestr("ignore.txt", "not collected")

    documents = collect_documents_with_docling([str(zip_path)])

    assert len(documents) == 1
    assert documents[0].source_type == "pdf"
    assert documents[0].source_url.endswith("#notice.pdf")


def test_collect_documents_with_docling_skips_failed_conversions(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: _FailingConverter(),
    )

    documents = collect_documents_with_docling(["https://example.com/file.pdf"])

    assert documents == []


def test_collect_documents_with_docling_uses_pdf_text_before_docling(
    monkeypatch, tmp_path
) -> None:
    converter = _CountingConverter()
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: converter,
    )
    monkeypatch.setattr(
        "app.crawlers.docling_collector._extract_pdf_text",
        lambda path: "PDF text content",
    )
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF")

    documents = collect_documents_with_docling([str(pdf_path)])

    assert len(documents) == 1
    assert documents[0].content == "PDF text content"
    assert converter.calls == 0


def test_collect_documents_with_docling_falls_back_to_docling_for_scanned_pdf(
    monkeypatch, tmp_path
) -> None:
    converter = _CountingConverter()
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: converter,
    )
    monkeypatch.setattr(
        "app.crawlers.docling_collector._extract_pdf_text",
        lambda path: "",
    )
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")

    documents = collect_documents_with_docling([str(pdf_path)])

    assert len(documents) == 1
    assert documents[0].content == "# Fallback\ncontent"
    assert converter.calls == 1


def test_collect_documents_with_docling_uses_page_ocr_before_docling(
    monkeypatch, tmp_path
) -> None:
    converter = _CountingConverter()
    monkeypatch.setattr(
        "app.crawlers.docling_collector._create_converter",
        lambda: converter,
    )
    monkeypatch.setattr(
        "app.crawlers.docling_collector._extract_pdf_text",
        lambda path: "",
    )
    monkeypatch.setattr(
        "app.crawlers.docling_collector._extract_pdf_ocr_text",
        lambda path, max_pages, scale: "OCR text content",
    )
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")

    documents = collect_documents_with_docling([str(pdf_path)])

    assert len(documents) == 1
    assert documents[0].content == "OCR text content"
    assert converter.calls == 0
