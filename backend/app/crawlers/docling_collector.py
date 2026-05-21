import hashlib
import re
import signal
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree
import zlib

import requests
from bs4 import BeautifulSoup

from app.schemas import Document

HWP_EXTENSIONS = {".hwp", ".hwpx"}
ARCHIVE_EXTENSIONS = {".zip"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
HTML_EXTENSIONS = {".html", ".htm"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
DOCLING_EXTENSIONS = {".pdf", ".docx", *IMAGE_EXTENSIONS, *HTML_EXTENSIONS, *MARKDOWN_EXTENSIONS}
SUPPORTED_DOWNLOAD_SUFFIXES = DOCLING_EXTENSIONS | HWP_EXTENSIONS | ARCHIVE_EXTENSIONS
DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}


@dataclass
class DoclingCollectorConfig:
    category: Optional[str] = None
    department: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    skip_unsupported: bool = False
    skip_images: bool = False
    skip_failed_conversions: bool = True
    prefer_pdf_text_extraction: bool = True
    pdf_text_max_pages: int = 30
    enable_pdf_page_ocr: bool = True
    pdf_ocr_max_pages: int = 30
    pdf_ocr_scale: float = 1.0
    timeout_seconds: int = 10
    conversion_timeout_seconds: int = 45
    max_zip_member_bytes: int = 25 * 1024 * 1024


def collect_documents_with_docling(
    sources: Iterable[str],
    config: Optional[DoclingCollectorConfig] = None,
) -> List[Document]:
    """Convert file paths or URLs into normalized documents using Docling."""
    config = config or DoclingCollectorConfig()
    converter = None
    collected_at = datetime.now()
    documents: List[Document] = []

    for source in sources:
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            local_source = _materialize_source(source, config, Path(temp_dir.name))
            source_type = _infer_source_type(
                source=local_source,
                headers=config.headers,
                timeout_seconds=config.timeout_seconds,
            )
            if source_type == "image" and config.skip_images:
                continue
            if source_type == "file":
                continue
            if source_type == "zip":
                documents.extend(
                    _collect_zip_documents(
                        archive_path=Path(local_source),
                        source=source,
                        config=config,
                        collected_at=collected_at,
                        temp_root=Path(temp_dir.name),
                    )
                )
                continue
            if source_type in {"hwp", "hwpx"}:
                content = _extract_hwp_content(Path(local_source), source_type)
            elif source_type == "pdf" and config.prefer_pdf_text_extraction:
                content = _run_with_timeout(
                    _extract_pdf_text_compat,
                    Path(local_source),
                    timeout_seconds=config.conversion_timeout_seconds,
                    max_pages=config.pdf_text_max_pages,
                )
                if not content and config.enable_pdf_page_ocr:
                    content = _run_with_timeout(
                        _extract_pdf_ocr_text_compat,
                        Path(local_source),
                        timeout_seconds=config.conversion_timeout_seconds,
                        max_pages=config.pdf_ocr_max_pages,
                        scale=config.pdf_ocr_scale,
                    )
                if not content:
                    converter = converter or _create_converter()
                    result = _run_with_timeout(
                        converter.convert,
                        local_source,
                        timeout_seconds=config.conversion_timeout_seconds,
                        headers=_request_headers(config.headers),
                    )
                    content = result.document.export_to_markdown().strip()
            else:
                converter = converter or _create_converter()
                result = _run_with_timeout(
                    converter.convert,
                    local_source,
                    timeout_seconds=config.conversion_timeout_seconds,
                    headers=_request_headers(config.headers),
                )
                content = result.document.export_to_markdown().strip()
        except ValueError:
            if config.skip_unsupported:
                continue
            raise
        except requests.RequestException:
            continue
        except Exception:
            if config.skip_failed_conversions:
                continue
            raise
        finally:
            temp_dir.cleanup()

        if not content:
            continue
        title = _extract_title(
            content=content,
            source=source,
            source_type=source_type,
            headers=config.headers,
            timeout_seconds=config.timeout_seconds,
        )

        documents.append(
            Document(
                doc_id=_build_doc_id(source),
                source_type=source_type,
                source_url=source,
                title=title,
                content=content,
                category=config.category,
                department=config.department,
                author_department=config.department,
                published_at=None,
                attachment_urls=[],
                collected_at=collected_at,
            )
        )

    return documents


def _create_converter():
    try:
        from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Add 'docling' to dependencies and install it first."
        ) from exc

    pdf_options = PdfPipelineOptions()
    pdf_options.ocr_options = RapidOcrOptions()
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        },
    )


def _run_with_timeout(func, *args, timeout_seconds: int, **kwargs):
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return func(*args, **kwargs)

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"{func.__name__} exceeded {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _materialize_source(source: str, config: DoclingCollectorConfig, temp_dir: Path) -> str:
    if not _is_url(source):
        return source

    suffix = Path(urlparse(source).path).suffix.lower()
    if suffix in DOCLING_EXTENSIONS and "downloadbbsfile.do" not in source.lower():
        return source

    response = requests.get(
        source,
        timeout=config.timeout_seconds,
        headers=_request_headers(config.headers),
        stream=True,
    )
    response.raise_for_status()

    filename = _filename_from_headers(response.headers) or Path(urlparse(source).path).name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOWNLOAD_SUFFIXES:
        suffix = _suffix_from_content_type(
            response.headers.get("Content-Type", "")
        )
    if suffix not in SUPPORTED_DOWNLOAD_SUFFIXES:
        suffix = ""
    local_path = temp_dir / f"download{suffix or '.bin'}"
    first_bytes = b""
    with local_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                if not first_bytes:
                    first_bytes = chunk[:64]
                file.write(chunk)

    if not suffix:
        detected_suffix = _suffix_from_file_signature(first_bytes)
        if detected_suffix:
            detected_path = local_path.with_suffix(detected_suffix)
            local_path.replace(detected_path)
            local_path = detected_path
    return str(local_path)


def _suffix_from_file_signature(data: bytes) -> str:
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK\x03\x04"):
        return ".zip"
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return ".hwp"
    if data.lstrip().startswith((b"<!doctype html", b"<html")):
        return ".html"
    return ""


def _filename_from_headers(headers: Dict[str, str]) -> Optional[str]:
    disposition = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not disposition:
        return None
    encoded_match = re.search(r"filename\*=UTF-8''([^;]+)", disposition, flags=re.IGNORECASE)
    if encoded_match:
        return unquote(encoded_match.group(1).strip().strip('"'))
    filename_match = re.search(r'filename="?([^";]+)"?', disposition, flags=re.IGNORECASE)
    if filename_match:
        return unquote(filename_match.group(1).strip())
    return None


def _suffix_from_content_type(content_type: str) -> str:
    normalized = content_type.lower()
    if "pdf" in normalized:
        return ".pdf"
    if "wordprocessingml.document" in normalized or "msword" in normalized:
        return ".docx"
    if "zip" in normalized:
        return ".zip"
    return ""


def _infer_source_type(
    source: str,
    headers: Dict[str, str],
    timeout_seconds: int,
) -> str:
    lowered_source = source.lower()
    if ".pdf" in lowered_source:
        return "pdf"
    if ".docx" in lowered_source:
        return "docx"

    suffix = Path(urlparse(source).path).suffix.lower()

    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix == ".hwp":
        return "hwp"
    if suffix == ".hwpx":
        return "hwpx"
    if suffix == ".zip":
        return "zip"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in HTML_EXTENSIONS:
        return "html"
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"

    content_type = _detect_content_type(
        source=source,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    if not content_type:
        if _is_url(source):
            return "html"
        return "file"

    normalized = content_type.lower()
    if "pdf" in normalized:
        return "pdf"
    if "wordprocessingml.document" in normalized or "msword" in normalized:
        return "docx"
    if "hwp" in normalized:
        return "hwp"
    if "zip" in normalized:
        return "zip"
    if normalized.startswith("image/"):
        return "image"
    if "html" in normalized:
        return "html"
    if "markdown" in normalized or "text/plain" in normalized:
        return "markdown"

    return "file"


def _collect_zip_documents(
    archive_path: Path,
    source: str,
    config: DoclingCollectorConfig,
    collected_at: datetime,
    temp_root: Path,
) -> List[Document]:
    documents: List[Document] = []
    extract_dir = temp_root / "zip"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            suffix = member_path.suffix.lower()
            if suffix not in DOCLING_EXTENSIONS | HWP_EXTENSIONS | ARCHIVE_EXTENSIONS:
                continue
            if member.file_size > config.max_zip_member_bytes:
                continue
            target_path = _safe_zip_extract_path(extract_dir, member_path.name)
            with archive.open(member) as source_file, target_path.open("wb") as target_file:
                target_file.write(source_file.read())
            nested_documents = collect_documents_with_docling(
                [str(target_path)],
                config=config,
            )
            for document in nested_documents:
                document.source_url = f"{source}#{member.filename}"
                document.doc_id = _build_doc_id(document.source_url)
                document.collected_at = collected_at
            documents.extend(nested_documents)

    return documents


def _safe_zip_extract_path(extract_dir: Path, filename: str) -> Path:
    target_path = extract_dir / Path(filename).name
    resolved_dir = extract_dir.resolve()
    resolved_target = target_path.resolve()
    if resolved_dir not in resolved_target.parents and resolved_target != resolved_dir:
        raise ValueError("Unsafe zip member path.")
    return target_path


def _extract_hwp_content(path: Path, source_type: str) -> str:
    if source_type == "hwpx":
        return _extract_hwpx_text(path)
    return _extract_hwp_text(path)


def _extract_hwpx_text(path: Path) -> str:
    lines: List[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            lowered = name.lower()
            if not lowered.endswith(".xml"):
                continue
            if not any(token in lowered for token in ("section", "contents", "body")):
                continue
            try:
                root = ElementTree.fromstring(archive.read(name))
            except ElementTree.ParseError:
                continue
            for element in root.iter():
                text = (element.text or "").strip()
                if text:
                    lines.append(text)
    return "\n".join(lines)


def _extract_hwp_text(path: Path) -> str:
    try:
        import olefile
    except ImportError as exc:
        raise ValueError("Install 'olefile' to extract HWP attachments.") from exc

    lines: List[str] = []
    with olefile.OleFileIO(str(path)) as ole:
        compressed = _is_hwp_compressed(ole)
        for stream_name in sorted(ole.listdir(streams=True)):
            if len(stream_name) < 2 or stream_name[0] != "BodyText":
                continue
            data = ole.openstream(stream_name).read()
            if compressed:
                data = zlib.decompress(data, -15)
            text = data.decode("utf-16le", errors="ignore")
            cleaned = _clean_hwp_text(text)
            if cleaned:
                lines.append(cleaned)
    return "\n".join(lines)


def _is_hwp_compressed(ole) -> bool:
    try:
        header = ole.openstream("FileHeader").read()
    except Exception:
        return False
    return len(header) > 36 and bool(header[36] & 0x01)


def _clean_hwp_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"그림입니다\.", " ", text)
    text = re.sub(
        r"원본 그림의 이름\s*:\s*.*?(?=원본 그림의 크기|프로그램 이름|$)",
        " ",
        text,
    )
    text = re.sub(
        r"원본 그림의 크기\s*:\s*.*?(?=프로그램 이름|$)",
        " ",
        text,
    )
    text = re.sub(
        r"프로그램 이름\s*:\s*[A-Za-z0-9 ._-]+",
        " ",
        text,
    )
    text = re.sub(
        r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,:;!?()\[\]{}<>/%~+\-=·ㆍ…'\"“”‘’→←↔①-⑳一-龥]",
        " ",
        text,
    )
    text = re.sub(
        r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,:;!?()\[\]{}<>/%~+\-=·ㆍ…'\"“”‘’→←↔①-⑳]",
        " ",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    lines = [
        line
        for line in (_clean_hwp_line(line) for line in text.splitlines())
        if line and _looks_like_useful_text(line)
    ]
    return _clean_hwp_text("\n".join(lines))


def _clean_hwp_line(value: str) -> str:
    line = _normalize_extracted_text_line(value)
    if any(
        token in line
        for token in ("그림입니다", "원본 그림의", "프로그램 이름")
    ):
        return ""
    year_match = re.search(r"\d{4}[-.년]", line)
    if year_match:
        prefix = line[: year_match.start()]
        prefix_tokens = re.findall(r"[0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+", prefix)
        short_prefix_tokens = [token for token in prefix_tokens if len(token) <= 2]
        mostly_short_noise = bool(prefix_tokens) and (
            len(short_prefix_tokens) / len(prefix_tokens) >= 0.6
        )
        if ":" not in prefix and (mostly_short_noise or not _looks_like_useful_text(prefix)):
            line = line[year_match.start() :]
    return re.sub(r"^(?:[A-Za-z가-힣]\s+){1,6}(?=\d{4})", "", line).strip()


def _clean_hwp_text(text: str) -> str:
    text = _fix_extracted_text(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    lines = [
        line
        for line in (_clean_hwp_line(line) for line in text.splitlines())
        if line and _looks_like_useful_text(line)
    ]
    return "\n".join(lines)


def _fix_extracted_text(text: str) -> str:
    try:
        from ftfy import fix_text
    except ImportError:
        return text
    return fix_text(text)


def _clean_hwp_line(value: str) -> str:
    line = _normalize_extracted_text_line(value)
    if _looks_like_image_metadata(line):
        return ""

    line = re.sub(
        r"[^0-9A-Za-z\uac00-\ud7a3\u3131-\u318e\s.,:;!?()\[\]{}<>/%~+\-=\u2190-\u2194]",
        " ",
        line,
    )
    line = _normalize_extracted_text_line(line)
    if _looks_like_short_token_noise(line):
        return ""

    year_match = re.search(r"\d{4}[-.\ub144]", line)
    if year_match:
        prefix = line[: year_match.start()]
        prefix_tokens = re.findall(r"[0-9A-Za-z\uac00-\ud7a3\u3131-\u318e]+", prefix)
        short_prefix_tokens = [token for token in prefix_tokens if len(token) <= 2]
        mostly_short_noise = bool(prefix_tokens) and (
            len(short_prefix_tokens) / len(prefix_tokens) >= 0.6
        )
        if ":" not in prefix and (mostly_short_noise or not _looks_like_useful_text(prefix)):
            line = line[year_match.start() :]

    return re.sub(
        r"^(?:[A-Za-z\uac00-\ud7a3\u3131-\u318e]\s+){1,8}(?=\d{4})",
        "",
        line,
    ).strip()


def _looks_like_short_token_noise(line: str) -> bool:
    tokens = re.findall(r"[0-9A-Za-z\uac00-\ud7a3\u3131-\u318e]+", line)
    if any(char.isdigit() for char in line):
        return False
    short_tokens = [token for token in tokens if len(token) <= 2]
    if len(tokens) < 3:
        return bool(tokens) and len(short_tokens) == len(tokens) and len(line) <= 10
    return len(short_tokens) / len(tokens) >= 0.8


def _looks_like_image_metadata(line: str) -> bool:
    lowered = line.lower()
    return any(
        token in lowered
        for token in (
            "\uadf8\ub9bc\uc785\ub2c8\ub2e4",
            "\uc6d0\ubcf8 \uadf8\ub9bc\uc758",
            "\ud504\ub85c\uadf8\ub7a8 \uc774\ub984",
            "imageready",
        )
    )


def _extract_pdf_text(path: Path, max_pages: int) -> str:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ""

    lines: List[str] = []
    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception:
        return ""

    try:
        page_count = min(len(pdf), max_pages)
        for page_index in range(page_count):
            try:
                page = pdf[page_index]
                text_page = page.get_textpage()
                text = text_page.get_text_range().strip()
            except Exception:
                continue
            finally:
                try:
                    text_page.close()
                except Exception:
                    pass
            if text:
                lines.append(text)
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return "\n\n".join(lines)


def _extract_pdf_text_compat(path: Path, max_pages: int) -> str:
    try:
        return _extract_pdf_text(path, max_pages)
    except TypeError:
        return _extract_pdf_text(path)  # type: ignore[call-arg]


def _extract_pdf_ocr_text(path: Path, max_pages: int, scale: float) -> str:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ""

    lines: List[str] = []
    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception:
        return ""

    try:
        ocr = _get_rapidocr()
        page_count = min(len(pdf), max_pages)
        for page_index in range(page_count):
            try:
                page = pdf[page_index]
                bitmap = page.render(scale=scale)
                image = bitmap.to_numpy()
                result = ocr(image)
                page_text = _rapidocr_result_to_text(result)
                if page_text:
                    lines.append(page_text)
            except Exception:
                continue
            finally:
                for resource_name in ("bitmap", "page"):
                    resource = locals().get(resource_name)
                    try:
                        resource.close()
                    except Exception:
                        pass
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return "\n\n".join(lines)


def _extract_pdf_ocr_text_compat(path: Path, max_pages: int, scale: float) -> str:
    try:
        return _extract_pdf_ocr_text(path, max_pages, scale)
    except TypeError:
        return _extract_pdf_ocr_text(path, max_pages)  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def _get_rapidocr():
    from rapidocr import RapidOCR

    return RapidOCR()


def _rapidocr_result_to_text(result) -> str:
    markdown = getattr(result, "to_markdown", None)
    if callable(markdown):
        text = markdown()
        if text:
            return text.strip()
    texts = getattr(result, "txts", None)
    if texts:
        return "\n".join(text for text in texts if str(text).strip())
    return ""


def _extract_title(
    content: str,
    source: str,
    source_type: str,
    headers: Dict[str, str],
    timeout_seconds: int,
) -> str:
    if source_type == "html":
        html_title = _extract_html_title(
            source=source,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        if html_title:
            return html_title[:300]

    for line in content.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        plain_text = re.sub(r"^#+\s*", "", normalized).strip()
        if plain_text and _should_skip_title_candidate(plain_text):
            continue
        if normalized.startswith("#"):
            title = plain_text
            if title:
                return title[:300]
        return normalized[:300]

    fallback = Path(urlparse(source).path).stem or source
    return fallback[:300]


def _extract_html_title(
    source: str,
    headers: Dict[str, str],
    timeout_seconds: int,
) -> Optional[str]:
    try:
        if _is_url(source):
            response = requests.get(
                source,
                timeout=timeout_seconds,
                headers=_request_headers(headers),
            )
            response.raise_for_status()
            html = response.text
        else:
            html = Path(source).read_text(encoding="utf-8")
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")

    og_title = soup.select_one('meta[property="og:title"]')
    if og_title and og_title.get("content"):
        normalized = _normalize_title_text(og_title["content"])
        if normalized and not _looks_like_navigation_text(normalized):
            return normalized

    if soup.title and soup.title.string:
        normalized = _normalize_title_text(soup.title.string)
        if normalized and not _looks_like_navigation_text(normalized):
            return normalized

    first_heading = soup.select_one("h1")
    if first_heading:
        normalized = _normalize_title_text(first_heading.get_text(" ", strip=True))
        if normalized and not _looks_like_navigation_text(normalized):
            return normalized

    return None


def _normalize_title_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_navigation_text(value: str) -> bool:
    lowered = value.lower()
    navigation_tokens = {"주메뉴", "메뉴", "home", "login", "사이트맵", "검색"}
    return any(token in lowered for token in navigation_tokens)


def _should_skip_title_candidate(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True
    if _looks_like_navigation_text(normalized):
        return True
    if normalized.startswith(("- [", "* [")):
        return True
    if re.fullmatch(r"\[.+\]\(.+\)", normalized):
        return True
    if re.fullmatch(r"-\s*\[.+\]\(.+\)", normalized):
        return True
    if not _looks_like_useful_text(normalized):
        return True
    return False


def _normalize_extracted_text_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_useful_text(value: str) -> bool:
    normalized = value.strip()
    if len(normalized) < 2:
        return False

    meaningful = re.findall(r"[0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]", normalized)
    if not meaningful:
        return False

    return len(meaningful) / max(len(normalized), 1) >= 0.35


def _looks_like_useful_text(value: str) -> bool:
    normalized = value.strip()
    if len(normalized) < 2:
        return False

    meaningful = re.findall(r"[0-9A-Za-z\uac00-\ud7a3\u3131-\u318e]", normalized)
    if not meaningful:
        return False

    return len(meaningful) / max(len(normalized), 1) >= 0.35


def _detect_content_type(
    source: str,
    headers: Dict[str, str],
    timeout_seconds: int,
) -> Optional[str]:
    if not _is_url(source):
        return None

    request_headers = _request_headers(headers)

    try:
        response = requests.head(
            source,
            timeout=timeout_seconds,
            headers=request_headers,
            allow_redirects=True,
        )
        content_type = response.headers.get("Content-Type")
        if content_type:
            return content_type
    except Exception:
        pass

    try:
        response = requests.get(
            source,
            timeout=timeout_seconds,
            headers=request_headers,
            stream=True,
        )
        content_type = response.headers.get("Content-Type")
        response.close()
        return content_type
    except Exception:
        return None


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _request_headers(headers: Dict[str, str]) -> Dict[str, str]:
    request_headers = dict(DEFAULT_REQUEST_HEADERS)
    request_headers.update(headers or {})
    return request_headers


def _build_doc_id(source: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"doc-{digest}"
