def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    clean_text = text.strip()
    if not clean_text:
        return []

    chunks = []
    start = 0

    while start < len(clean_text):
        end = start + chunk_size
        chunks.append(clean_text[start:end])

        if end >= len(clean_text):
            break
        start = end - chunk_overlap

    return chunks

