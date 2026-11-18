#!/usr/bin/env python3
"""Verify and restore the nine-task results archive without running experiments.

Requires Python 3.12 or newer; uses only the standard library.

From the downloaded results branch:
    python restore.py --verify-only
    python restore.py C:\\TADA-restored

The destination must not exist. Its parent directory must already exist.
All compressed parts are verified before creating the destination. Archives are
then read as streams, so neither a joined archive nor a full-memory copy is made.
Windows paths longer than 260 characters are handled automatically.
If extraction fails, the partial destination is preserved for inspection; choose
a different new destination after resolving the error. This script never deletes
existing files, rewrites historical records, or starts an experiment.

Advanced: --manifest PATH selects another manifest. Part paths are interpreted
relative to the parent of its archives directory (the downloaded branch root).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tarfile


CHUNK_SIZE = 1024 * 1024
PART_LIMIT = 48 * 1024 * 1024


class RestoreError(Exception):
    """An invalid manifest, unsafe archive, or failed integrity check."""


def filesystem_path(path):
    """Use one normalized absolute path for checks and long-path file access."""
    absolute = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        # Normalize before adding the extended prefix: Windows deliberately
        # stops normalizing dot segments and separators after that prefix.
        absolute = os.path.normpath(absolute)
        absolute = ("\\\\?\\UNC\\" + absolute[2:]
                    if absolute.startswith("\\\\") else "\\\\?\\" + absolute)
    return Path(absolute)


def load_manifest(manifest_path: Path):
    manifest_path = filesystem_path(manifest_path).resolve(strict=True)
    root = manifest_path.parent.parent
    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or type(value.get("format")) is not int or value["format"] != 1:
        raise RestoreError("Unsupported manifest format; expected integer format 1.")
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RestoreError("Manifest must contain at least one archive group.")
    names, paths, prepared = set(), set(), []
    for group in groups:
        if not isinstance(group, dict):
            raise RestoreError("Each archive group must be an object.")
        name = group.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise RestoreError("Archive group names must be nonempty and unique.")
        names.add(name)
        if group.get("archive_format") != "tar.gz":
            raise RestoreError(f"Unsupported archive format in group {name!r}.")
        parts = group.get("parts")
        if not isinstance(parts, list) or not parts:
            raise RestoreError(f"Group {name!r} has no parts.")
        prepared_parts = []
        for part in parts:
            if not isinstance(part, dict):
                raise RestoreError(f"Invalid part in group {name!r}.")
            relative = part.get("path")
            if not isinstance(relative, str) or not relative:
                raise RestoreError("Every part must have a relative path.")
            posix = PurePosixPath(relative)
            windows = PureWindowsPath(relative)
            if (posix.is_absolute() or windows.drive or "\\" in relative
                    or ".." in posix.parts or not posix.parts or posix.parts[0] != "archives"):
                raise RestoreError(f"Unsafe part path: {relative!r}")
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise RestoreError(f"Part is not a regular file inside the archive package: {relative!r}")
            if path in paths:
                raise RestoreError(f"Part appears more than once: {relative!r}")
            paths.add(path)
            size, digest = part.get("bytes"), part.get("sha256")
            if type(size) is not int or not 0 < size <= PART_LIMIT:
                raise RestoreError(f"Part size must be from 1 to 48 MiB: {relative!r}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise RestoreError(f"Invalid SHA256 for {relative!r}")
            prepared_parts.append({"path": path, "label": relative,
                                   "bytes": size, "sha256": digest.lower()})
        prepared.append({"name": name, "parts": prepared_parts})
    return prepared


def verify_parts(groups):
    total_parts = sum(len(group["parts"]) for group in groups)
    total_bytes = sum(part["bytes"] for group in groups for part in group["parts"])
    number = 0
    for group in groups:
        for part in group["parts"]:
            number += 1
            path = part["path"]
            if path.stat().st_size != part["bytes"]:
                raise RestoreError(f"Size mismatch: {part['label']}")
            digest = hashlib.sha256()
            observed = 0
            with path.open("rb") as stream:
                while chunk := stream.read(CHUNK_SIZE):
                    observed += len(chunk)
                    digest.update(chunk)
            if observed != part["bytes"] or digest.hexdigest() != part["sha256"]:
                raise RestoreError(f"SHA256 or size mismatch: {part['label']}")
            print(f"Verified {number}/{total_parts}: {part['label']}", flush=True)
    print(f"Integrity verified: {total_parts} parts, {total_bytes:,} compressed bytes.", flush=True)


class JoinedParts(io.RawIOBase):
    """A sequential reader that checks each part again as extraction consumes it."""

    def __init__(self, parts):
        super().__init__()
        self.parts = iter(parts)
        self.current = None
        self.part = None
        self.digest = None
        self.observed = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        if not view:
            return 0
        while True:
            if self.current is None:
                self.part = next(self.parts, None)
                if self.part is None:
                    return 0
                self.current = self.part["path"].open("rb")
                self.digest = hashlib.sha256()
                self.observed = 0
            count = self.current.readinto(view)
            if count:
                self.digest.update(view[:count])
                self.observed += count
                return count
            self.current.close()
            self.current = None
            if (self.observed != self.part["bytes"]
                    or self.digest.hexdigest() != self.part["sha256"]):
                raise RestoreError(f"Part changed during extraction: {self.part['label']}")

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def safe_members(archive):
    for member in archive:
        # Reject portable pathname ambiguities before tarfile's data filter checks
        # destination paths, link targets, file types, ownership and permissions.
        name = member.name
        posix = PurePosixPath(name)
        if (posix.is_absolute() or PureWindowsPath(name).drive
                or ".." in posix.parts or "\\" in name):
            raise RestoreError(f"Unsafe archive member path: {name!r}")
        yield member


def restore_groups(groups, destination: Path):
    destination = filesystem_path(destination)
    if destination.exists() or destination.is_symlink():
        raise RestoreError(f"Destination already exists; choose a new directory: {destination}")
    if not destination.parent.is_dir():
        raise RestoreError(f"Destination parent does not exist: {destination.parent}")
    destination.mkdir(exist_ok=False)
    for group in groups:
        print(f"Extracting: {group['name']}", flush=True)
        with JoinedParts(group["parts"]) as raw:
            with io.BufferedReader(raw, buffer_size=CHUNK_SIZE) as stream:
                with tarfile.open(fileobj=stream, mode="r|gz", errorlevel=2) as archive:
                    archive.extractall(path=destination, members=safe_members(archive), filter="data")
                # Tar ends before gzip padding in some archives. Consume the
                # entire compressed stream to finish every per-part recheck.
                while stream.read(CHUNK_SIZE):
                    pass
    print(f"Restored to: {destination}\nNo experiments were started.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("destination", nargs="?", type=Path, help="new, nonexistent restore directory")
    parser.add_argument("--verify-only", action="store_true", help="verify all part hashes and sizes without extraction")
    parser.add_argument("--manifest", type=Path,
                        default=filesystem_path(__file__).resolve().parent / "archives" / "manifest.json")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required for safe tar extraction.")
    if args.verify_only and args.destination is not None:
        parser.error("Do not supply a destination with --verify-only.")
    if not args.verify_only and args.destination is None:
        parser.error("Supply a new destination directory, or use --verify-only.")
    destination = filesystem_path(args.destination) if args.destination is not None else None
    try:
        if destination is not None and (destination.exists() or destination.is_symlink()):
            raise RestoreError(f"Destination already exists; choose a new directory: {destination}")
        groups = load_manifest(args.manifest)
        verify_parts(groups)
        if not args.verify_only:
            restore_groups(groups, destination)
    except (RestoreError, OSError, ValueError, tarfile.TarError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        if destination is not None and destination.exists():
            print("Any partial destination was preserved; inspect it before trying a new destination.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
