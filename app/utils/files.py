import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", name)
    return name.strip(" .")


def is_safe_child_path(file_path: Path, parent_dir: Path) -> bool:
    try:
        file_path.resolve().relative_to(parent_dir.resolve())
    except ValueError:
        return False
    return True

