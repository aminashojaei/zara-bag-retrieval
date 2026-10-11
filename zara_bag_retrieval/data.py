import csv
from dataclasses import dataclass
from pathlib import Path

from .config import IMAGE_SUFFIXES


@dataclass(frozen=True)
class ImageRecord:
    product_id: str
    path: Path
    relative_key: str


def scan_dataset_role(dataset_root: Path, role: str) -> list[ImageRecord]:
    root = Path(dataset_root) / role
    if role not in ("query", "gallery"):
        raise ValueError(f"Unsupported role: {role}")
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {root.resolve()}")
    rows = []
    for product in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not product.is_dir():
            continue
        for file in sorted(product.rglob("*"), key=lambda p: str(p).casefold()):
            if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(ImageRecord(product.name, file.resolve(), file.relative_to(root).as_posix()))
    return rows


def read_inventory(path: Path, dataset_root: Path, role: str) -> list[ImageRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing inventory: {path}")
    result = []
    anchor = f"/{role}/"
    role_root = (Path(dataset_root) / role).resolve()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"product_id", "path"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Inventory must contain product_id,path: {path}")
        for line_number, row in enumerate(reader, 2):
            source = row["path"].replace("\\", "/")
            pos = ("/" + source).lower().rfind(anchor)
            if pos < 0:
                raise ValueError(f"Inventory row {line_number} missing /{role}/ in path: {source}")
            relative_key = ("/" + source)[pos + len(anchor):]
            parts = Path(relative_key).parts
            if len(parts) < 2 or ".." in parts or parts[0] != row["product_id"]:
                raise ValueError(f"Invalid inventory row {line_number}: {row}")
            file = (role_root / relative_key).resolve()
            if not file.is_relative_to(role_root):
                raise ValueError("Refusing a path outside the dataset")
            result.append(ImageRecord(row["product_id"], file, relative_key))
    return result


def assert_inventory_matches(actual: list[ImageRecord], cached: list[ImageRecord], label: str) -> None:
    a = [(r.product_id, r.relative_key) for r in actual]
    b = [(r.product_id, r.relative_key) for r in cached]
    if a != b:
        raise ValueError(
            f"{label} has changed since embeddings were generated. "
            "Restore the exact query/gallery split or run python build_artifacts.py --force "
            "before launching the app."
        )
    missing = [str(r.path) for r in cached if not r.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {label}: {missing[:3]}")
