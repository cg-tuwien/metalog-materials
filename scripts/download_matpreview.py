# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

"""Download the Mitsuba "matpreview" scene and extract it into scenes/matpreview/.

The scene geometry and environment map are not redistributed with this repository;
only the project-specific scene XMLs live in scenes/matpreview/. This script fetches
the remaining files from the Mitsuba scene repository.

Uses only the Python standard library, so it can be run on the host without the
project environment.

Examples:
    python scripts/download_matpreview.py            # download missing files
    python scripts/download_matpreview.py --force    # overwrite existing files
    python scripts/download_matpreview.py --check    # compare archive with local files, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://d38rqfq1h7iukm.cloudfront.net/scenes/matpreview.zip"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "scenes" / "matpreview"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def archive_members(archive: zipfile.ZipFile) -> dict[PurePosixPath, zipfile.ZipInfo]:
    """Map relative output paths to zip entries, stripping a common 'matpreview/' prefix."""
    infos = [
        info
        for info in archive.infolist()
        if not info.is_dir() and not info.filename.startswith("__MACOSX/")
    ]
    paths = [PurePosixPath(info.filename) for info in infos]
    strip = all(len(p.parts) > 1 and p.parts[0] == "matpreview" for p in paths)

    members = {}
    for info, path in zip(infos, paths):
        rel = PurePosixPath(*path.parts[1:]) if strip else path
        if rel.is_absolute() or ".." in rel.parts:
            sys.exit(f"ERROR: unsafe path in archive: {info.filename}")
        members[rel] = info
    return members


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(archive: zipfile.ZipFile, members, out_dir: Path) -> int:
    differ = 0
    for rel, info in sorted(members.items()):
        dst = out_dir / rel
        if not dst.is_file():
            status = "missing"
            differ += 1
        elif sha256(dst.read_bytes()) == sha256(archive.read(info)):
            status = "identical"
        else:
            status = "DIFFERS"
            differ += 1
        print(f"{status:10} {rel}")

    archived = {out_dir / rel for rel in members}
    if out_dir.is_dir():
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            if path not in archived:
                print(f"{'local-only':10} {path.relative_to(out_dir).as_posix()}")
    return differ


def extract(archive: zipfile.ZipFile, members, out_dir: Path, force: bool):
    for rel, info in sorted(members.items()):
        dst = out_dir / rel
        if dst.is_file() and not force:
            print(f"skip      {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            tmp.write_bytes(archive.read(info))
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"done      {rel}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the archive with the files in --out-dir without writing anything",
    )
    args = parser.parse_args()

    print(f"Downloading {URL} ...", flush=True)
    archive = zipfile.ZipFile(io.BytesIO(fetch(URL)))
    members = archive_members(archive)

    if args.check:
        differ = check(archive, members, args.out_dir)
        sys.exit(1 if differ else 0)

    extract(archive, members, args.out_dir, args.force)
    print(f"All files available in {args.out_dir}")


if __name__ == "__main__":
    main()
