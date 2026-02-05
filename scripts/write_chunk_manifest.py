#!/usr/bin/env python3
"""
Create chunk files + a manifest.txt from a flat list of layout JSON paths.

This mirrors the chunking layout used by the Slurm export scripts:
  - <chunk_dir>/chunk_0000, chunk_0001, ...
  - <chunk_dir>/manifest.txt  (one chunk file path per line)

Example (on Greene):
  python scripts/write_chunk_manifest.py \
    --layouts-list /vast/.../unique_outputs_missing_dell/layouts.txt \
    --chunk-dir /scratch/sxr203/newspaper-parsing/chunks_missing_dell_100 \
    --chunk-size 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write chunk_* files + manifest.txt from a layouts list file.")
    ap.add_argument("--layouts-list", required=True, help="Path to layouts.txt (one JSON path per line)")
    ap.add_argument("--chunk-dir", required=True, help="Directory to write chunk files + manifest.txt into")
    ap.add_argument("--chunk-size", type=int, default=100, help="Number of layout paths per chunk file")
    ap.add_argument(
        "--refuse-if-exists",
        action="store_true",
        help="Fail if chunk-dir already exists and is non-empty",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    layouts_list = Path(args.layouts_list).expanduser()
    chunk_dir = Path(args.chunk_dir).expanduser()

    if not layouts_list.is_file():
        raise SystemExit(f"--layouts-list not found: {layouts_list}")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")

    lines = []
    for raw in layouts_list.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if not lines:
        raise SystemExit(f"No layout paths found in {layouts_list}")

    if chunk_dir.exists() and args.refuse_if_exists:
        if any(chunk_dir.iterdir()):
            raise SystemExit(f"Refusing to write into non-empty chunk-dir: {chunk_dir}")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = chunk_dir / "manifest.txt"
    # If re-running into an existing directory, be explicit by truncating.
    manifest_path.write_text("")

    chunk_size = int(args.chunk_size)
    num_chunks = (len(lines) + chunk_size - 1) // chunk_size

    chunk_paths: list[Path] = []
    for i in range(num_chunks):
        chunk_lines = lines[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = chunk_dir / f"chunk_{i:04d}"
        chunk_path.write_text("\n".join(chunk_lines) + "\n")
        chunk_paths.append(chunk_path)

    manifest_path.write_text("\n".join(str(p) for p in chunk_paths) + "\n")

    print(f"Done. layouts={len(lines)} chunk_size={chunk_size} chunks={len(chunk_paths)}")
    print(f"Chunk dir: {chunk_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

