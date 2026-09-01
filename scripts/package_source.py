#!/usr/bin/env python3
"""Create a deterministic, reviewable source ZIP from an explicit whitelist."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path

ARCHIVE_ROOT = "enterprise-memory-mlx"
ROOT_FILES = {
    ".gitignore",
    ".python-version",
    "LICENSE",
    "Makefile",
    "README.md",
    "VALIDATION.md",
    "pyproject.toml",
}
ROOT_DIRECTORIES = {".github", "docs", "knowledge", "scripts", "src", "tests"}
EXCLUDED_NAMES = {
    ".DS_Store",
    "__MACOSX",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".safetensors",
    ".gguf",
    ".bin",
    ".mlx",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_source_files(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    files: list[Path] = []
    for name in sorted(ROOT_FILES):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required source file missing: {path}")
        files.append(path)
    for directory in sorted(ROOT_DIRECTORIES):
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"required source directory missing: {base}")
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root)
            if _excluded(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"source archive refuses symlink: {relative}")
            if path.is_file():
                files.append(path)
    relative_paths = [path.relative_to(root) for path in files]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("source whitelist produced duplicate paths")
    return tuple(files)


def build_source_zip(root: Path, output_path: Path) -> dict:
    root = root.resolve()
    output_path = output_path.resolve()
    files = collect_source_files(root)
    payloads = {
        str(path.relative_to(root)): path.read_bytes()
        for path in files
    }
    manifest = {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "file_count": len(payloads),
        "files": {
            name: {
                "sha256": sha256_bytes(content),
                "bytes": len(content),
            }
            for name, content in sorted(payloads.items())
        },
        "exclusions": [
            ".venv",
            "artifacts",
            "dev",
            "dist",
            ".specstory",
            "knowledge/private",
            "models and weights",
            "credentials and local configuration",
            "caches, bytecode, egg metadata, and macOS resource files",
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, content in sorted(payloads.items()):
                source = root / name
                mode = stat.S_IMODE(source.stat().st_mode)
                _write_member(
                    archive,
                    f"{ARCHIVE_ROOT}/{name}",
                    content,
                    mode=mode,
                )
            _write_member(
                archive,
                f"{ARCHIVE_ROOT}/SOURCE_MANIFEST.json",
                manifest_bytes,
                mode=0o644,
            )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "path": str(output_path),
        "sha256": sha256_bytes(output_path.read_bytes()),
        "file_count": len(payloads),
        "manifest_sha256": sha256_bytes(manifest_bytes),
    }


def verify_source_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest_name = f"{ARCHIVE_ROOT}/SOURCE_MANIFEST.json"
        if names.count(manifest_name) != 1:
            raise ValueError("source archive must contain one SOURCE_MANIFEST.json")
        manifest = json.loads(archive.read(manifest_name))
        expected_names = {
            f"{ARCHIVE_ROOT}/{name}" for name in manifest.get("files", {})
        } | {manifest_name}
        if set(names) != expected_names or len(names) != len(set(names)):
            raise ValueError("source archive contents do not match its manifest")
        for relative, metadata in manifest["files"].items():
            content = archive.read(f"{ARCHIVE_ROOT}/{relative}")
            if sha256_bytes(content) != metadata["sha256"]:
                raise ValueError(f"source archive hash mismatch: {relative}")
            if len(content) != metadata["bytes"]:
                raise ValueError(f"source archive size mismatch: {relative}")
        forbidden = [name for name in names if _archive_name_forbidden(name)]
        if forbidden:
            raise ValueError(
                "source archive contains forbidden paths: " + ", ".join(forbidden)
            )
    return {
        "sha256": sha256_bytes(path.read_bytes()),
        "file_count": len(names) - 1,
        "verified": True,
    }


def _excluded(relative: Path) -> bool:
    parts = set(relative.parts)
    if parts & EXCLUDED_NAMES:
        return True
    if "private" in relative.parts and relative.parts[0] == "knowledge":
        return True
    if any(part.endswith(".egg-info") for part in relative.parts):
        return True
    if relative.suffix in EXCLUDED_SUFFIXES:
        return True
    return relative.name == ".env" or relative.name.startswith(".env.")


def _archive_name_forbidden(name: str) -> bool:
    relative = name.removeprefix(f"{ARCHIVE_ROOT}/")
    parts = Path(relative).parts
    forbidden_parts = {
        ".venv",
        "artifacts",
        "dev",
        "dist",
        ".specstory",
        "__MACOSX",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "private",
        "models",
    }
    return bool(set(parts) & forbidden_parts) or Path(relative).suffix in EXCLUDED_SUFFIXES


def _write_member(
    archive: zipfile.ZipFile,
    name: str,
    content: bytes,
    *,
    mode: int,
) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (mode & 0xFFFF) << 16
    info.create_system = 3
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "dist" / "enterprise-memory-mlx-source.zip"
    )
    result = build_source_zip(root, output)
    verification = verify_source_zip(output)
    print(
        json.dumps(
            {
                **result,
                "verified": verification["verified"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
