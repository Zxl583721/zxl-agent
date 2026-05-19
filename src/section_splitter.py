from pathlib import Path
import re

from src.text_splitter import split_text


CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千万\d]+[讲章节]\s*[\S ]{0,80}$"),
    re.compile(r"^\d+(?:\.\d+){0,4}[、.．]?\s+[\S ]{1,80}$"),
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*[\S ]{1,80}$"),
    re.compile(r"^（[一二三四五六七八九十\d]+）\s*[\S ]{1,80}$"),
    re.compile(r"^\([一二三四五六七八九十\d]+\)\s*[\S ]{1,80}$"),
]

PAGE_PREFIX_PATTERN = re.compile(r"^第\s*\d+\s*页$")
DEFAULT_CHAPTER = "未识别章节"


def detect_chapter_title(line: str) -> str | None:
    """Return a normalized chapter title when a line looks like a section heading."""
    candidate = normalize_line(line)
    if not candidate or PAGE_PREFIX_PATTERN.match(candidate):
        return None
    if len(candidate) > 90:
        return None

    for pattern in CHAPTER_PATTERNS:
        if pattern.match(candidate):
            return candidate

    return None


def group_pages_into_sections(pages: list[dict]) -> list[dict]:
    """Group page-level text into chapter-aware sections."""
    sections = []
    current = None

    for page in pages:
        source = page.get("source", "")
        filename = page.get("filename") or Path(source).name
        page_number = page.get("page")
        lines = [normalize_line(line) for line in str(page.get("text", "")).splitlines()]
        lines = [line for line in lines if line]

        for line in lines:
            if PAGE_PREFIX_PATTERN.match(line):
                continue

            title = detect_chapter_title(line)
            if title:
                if current and current["content_lines"]:
                    sections.append(finalize_section(current))
                current = {
                    "source": source,
                    "filename": filename,
                    "chapter": title,
                    "start_page": page_number,
                    "end_page": page_number,
                    "content_lines": [line],
                }
                continue

            if current is None or current["source"] != source:
                if current and current["content_lines"]:
                    sections.append(finalize_section(current))
                current = {
                    "source": source,
                    "filename": filename,
                    "chapter": DEFAULT_CHAPTER,
                    "start_page": page_number,
                    "end_page": page_number,
                    "content_lines": [],
                }

            current["end_page"] = page_number or current["end_page"]
            current["content_lines"].append(line)

    if current and current["content_lines"]:
        sections.append(finalize_section(current))

    return sections


def split_sections_into_chunks(
    sections: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split each section into chunks while carrying source, chapter, and page metadata."""
    chunks = []

    for section_index, section in enumerate(sections):
        text_chunks = split_text(
            section["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        safe_chapter = sanitize_id_part(section["chapter"])
        source_path = Path(section["source"])

        for chunk_index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "id": f"{source_path.name}-{section_index}-{safe_chapter}-{chunk_index}",
                    "text": text,
                    "metadata": {
                        "source": section["source"],
                        "filename": section["filename"],
                        "chapter": section["chapter"],
                        "start_page": section["start_page"] or "",
                        "end_page": section["end_page"] or "",
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def finalize_section(section: dict) -> dict:
    content = "\n".join(section["content_lines"]).strip()
    return {
        "source": section["source"],
        "filename": section["filename"],
        "chapter": section["chapter"],
        "start_page": section["start_page"],
        "end_page": section["end_page"],
        "content": content,
    }


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line).strip())


def sanitize_id_part(text: str) -> str:
    safe_text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text.strip())
    return safe_text[:60] or "section"
