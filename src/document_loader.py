from pathlib import Path


def load_txt_documents(data_dir: str | Path) -> list[dict]:
    """Load all .txt files from the knowledge base directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    documents = []
    for file_path in sorted(data_path.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gbk", errors="ignore").strip()

        if content:
            documents.append(
                {
                    "source": str(file_path),
                    "content": content,
                }
            )

    return documents


# TODO: 后续可以在这里扩展 PDF、Word、Markdown 等文件格式的加载函数。

