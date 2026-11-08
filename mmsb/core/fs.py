"""Files and folders: JSON that is written whole or not at all, deletion that works on Windows for read-only files and links, and
the frozen data folder with its manifest (moved from bench.py of the MultimediaSearch repo, fe0ece5)."""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path


def slug(name: str) -> str:
    """`name` as a file name: every character outside letters, digits, `_` and `-` becomes `-`."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


def write_text(path: Path, text: str) -> None:
    """`text` written to `path` whole or not at all (a `.part` file, then a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def write_json(path: Path, data) -> None:
    """`data` written to `path` whole or not at all."""
    write_text(path, json.dumps(data, indent=1))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    """The sha256 of the bytes of the file at `path`, as hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def kernel():
    """kernel32 with the signatures `unlink` uses (Windows)."""
    from ctypes import wintypes

    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.CreateFileW.restype = wintypes.HANDLE
    lib.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    lib.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    lib.CloseHandle.argtypes = [wintypes.HANDLE]
    return lib


def unlink(path: Path) -> None:
    """Delete a file, or a hardlink to one, without touching its attributes. On Windows a link to a read-only file cannot be
    deleted until the attribute is cleared, and clearing it clears it on every link (the snapshot's media, the run's data, the
    download): this deletes the name alone."""
    if os.name != "nt":
        os.unlink(path)
        return
    lib = kernel()
    handle = lib.CreateFileW(str(path), 0x00010000, 7, None, 3, 0x00200000, None)  # DELETE; share all; open the link itself
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        flags = ctypes.c_uint32(0x1 | 0x2 | 0x10)  # FileDispositionInfoEx: delete, POSIX semantics, ignore the read-only attribute
        if not lib.SetFileInformationByHandle(handle, 21, ctypes.byref(flags), 4):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        lib.CloseHandle(handle)


def remove_tree(path: Path) -> None:
    """`path` and all below it removed, a read-only file (the data, a file an exec copied) made writable first, except a file
    with other links (a hardlink an agent made, a home's media): that is deleted as a name (`unlink`), so the other links keep
    their attributes. A directory junction (an agent can make one with `mklink /J`) is removed as the link it is: `rmtree` refuses
    one it is given, and a junction met inside a tree is removed by it without going through."""
    if path.is_junction():
        os.rmdir(path)
        return

    def writable(function, target, error) -> None:
        if function is os.unlink and os.lstat(target).st_nlink > 1:
            unlink(Path(target))
            return
        os.chmod(target, stat.S_IWRITE)
        function(target)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=writable)
    elif path.lstat().st_nlink > 1:
        unlink(path)
    else:
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


def empty(folder: Path) -> None:
    """`folder` made, with nothing in it."""
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        remove_tree(child)


def stats_of(data: Path) -> dict[str, list[int]]:
    """[size, mtime in ns] of every file under `data`, by its path relative to `data` with forward slashes."""
    found = {}
    for path in sorted(data.rglob("*")):
        if path.is_file():
            info = path.stat()
            found[path.relative_to(data).as_posix()] = [info.st_size, info.st_mtime_ns]
    return found


def inherit_permissions(folder: Path) -> None:
    """On Windows, every file and folder under `folder` given back the permissions it inherits, and nothing of its own: a file
    with owner-only permissions (one moved out of a Python temporary folder) cannot be opened by Codex's sandbox, which reads
    as the CodexSandboxUsers group. A no-op elsewhere."""
    if os.name != "nt":
        return
    done = subprocess.run(["icacls", str(folder), "/reset", "/T", "/Q"], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"icacls /reset failed on {folder}: {(done.stderr or done.stdout).strip()}")


def own_permissions(folder: Path) -> list[str]:
    """On Windows, the files under `folder` with a permission of their own (not inherited), as paths relative to it, in one
    PowerShell `Get-Acl` pass; [] elsewhere."""
    if os.name != "nt":
        return []
    quoted = str(folder).replace("'", "''")
    # Stop, because a Get-Acl that fails would otherwise read as a file with nothing of its own. No PSModulePath, because a child of
    # PowerShell 7 (GitHub's runners) inherits one whose Get-Acl Windows PowerShell cannot load.
    script = ("$ErrorActionPreference = 'Stop'; "
              f"Get-ChildItem -LiteralPath '{quoted}' -Recurse -File | Where-Object {{ @((Get-Acl -LiteralPath $_.FullName).Access"
              " | Where-Object { -not $_.IsInherited }).Count -gt 0 } | ForEach-Object { $_.FullName }")
    env = {name: value for name, value in os.environ.items() if name.upper() != "PSMODULEPATH"}
    done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, env=env)
    if done.returncode:
        raise RuntimeError(f"Get-Acl failed on {folder}: {done.stderr.strip()}")
    return [Path(line.strip()).relative_to(folder).as_posix() for line in done.stdout.splitlines() if line.strip()]


def freeze(data: Path, manifest: Path) -> None:
    """Every file of `data` with only the permissions it inherits (so Codex's sandbox can read it) and read-only, and `manifest`
    (size and mtime of each) written. Raises when a file still has permissions of its own."""
    inherit_permissions(data)
    if mine := own_permissions(data):
        raise RuntimeError(f"{len(mine)} data files keep permissions of their own, which Codex's sandbox may not read: "
                           f"{', '.join(mine[:5])}")
    for path in data.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    manifest.write_text(json.dumps(stats_of(data)), encoding="utf-8")


def changed(data: Path, manifest: Path) -> str | None:
    """The first file of `data` that is gone, changed or added since `freeze` (`kind: path`), or None."""
    saved, now = read_json(manifest), stats_of(data)
    for rel, was in saved.items():
        if rel not in now:
            return f"gone: {rel}"
        if now[rel] != was:
            return f"changed: {rel}"
    return next((f"added: {rel}" for rel in now if rel not in saved), None)
