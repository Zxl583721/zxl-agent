import argparse
import json
from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 Chroma 中的 RAG child chunks，便于人工检查分块内容。")
    parser.add_argument("--persist-dir", default="vector_store", help="Chroma 持久化目录，例如 vector_store 或 vector_store/kb_3。")
    parser.add_argument("--collection", default="personal_knowledge_base", help="Chroma collection 名称，例如 personal_knowledge_base 或 kb_3。")
    parser.add_argument("--filename", help="只导出某个文件名包含该文本的 chunk。")
    parser.add_argument("--limit", type=int, default=20, help="最多导出多少个 chunk；0 表示全部。")
    parser.add_argument("--format", choices=("md", "jsonl"), default="md", help="导出格式。")
    parser.add_argument("--output", help="输出文件路径；不传则打印到终端。")
    return parser.parse_args()


def get_collection(persist_dir: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(collection_name)


def iter_chunks(collection, filename: str | None, limit: int):
    batch_size = 500
    exported = 0
    offset = 0

    while True:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", []) or []
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        if not ids:
            break

        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            document = documents[index] or ""
            if filename and filename not in str(metadata.get("filename", "")):
                continue

            yield {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
            }
            exported += 1
            if limit and exported >= limit:
                return

        offset += len(ids)


def format_markdown(chunks: list[dict]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        lines.extend(
            [
                f"## Chunk {index}",
                "",
                f"- id: `{chunk['id']}`",
                f"- filename: `{metadata.get('filename', '')}`",
                f"- chapter: `{metadata.get('chapter', '')}`",
                f"- pages: `{metadata.get('start_page', '')}` - `{metadata.get('end_page', '')}`",
                f"- parent_id: `{metadata.get('parent_id', '')}`",
                f"- section_index: `{metadata.get('section_index', '')}`",
                f"- chunk_index: `{metadata.get('chunk_index', '')}`",
                "",
                "```text",
                chunk["text"].strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def format_jsonl(chunks: list[dict]) -> str:
    return "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks)


def main() -> None:
    args = parse_args()
    persist_dir = Path(args.persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = BASE_DIR / persist_dir

    collection = get_collection(persist_dir, args.collection)
    chunks = list(iter_chunks(collection, args.filename, args.limit))
    output = format_jsonl(chunks) if args.format == "jsonl" else format_markdown(chunks)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"已导出 {len(chunks)} 个 chunk 到 {output_path}")
        return

    print(output)


if __name__ == "__main__":
    main()
