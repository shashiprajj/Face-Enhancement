"""Shared helpers: hashing, config loading, manifests, image statistics."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(rel: str | os.PathLike) -> Path:
    """Resolve a repo-relative path (all manifests store repo-relative POSIX paths)."""
    return REPO_ROOT / Path(rel)


def rel_path(p: str | os.PathLike) -> str:
    return Path(p).resolve().relative_to(REPO_ROOT).as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: str | os.PathLike) -> dict:
    with open(repo_path(path) if not Path(path).is_absolute() else path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = Path(path).as_posix()
    cfg["_config_sha256"] = sha256_file(repo_path(path) if not Path(path).is_absolute() else path)
    return cfg


def git_head() -> str:
    if os.environ.get("FES_GIT_HEAD"):
        return os.environ["FES_GIT_HEAD"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_info(extra: Mapping | None = None) -> dict:
    """Provenance block written next to every generated manifest."""
    info = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "git_head": git_head(),
    }
    if extra:
        info.update(extra)
    return info


def write_csv_gz(path: str | os.PathLike, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    # mtime=0 keeps the gzip bytes deterministic for identical content
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def read_csv_gz(path: str | os.PathLike) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str | os.PathLike, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


# ---------------------------------------------------------------- image statistics

def decode_image(data: bytes) -> np.ndarray | None:
    """Decode to BGR uint8 (3-channel). Returns None if undecodable."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def is_grayscale(img_bgr: np.ndarray, tol: int = 2) -> bool:
    """True if the three channels are (near) identical, i.e. a grayscale photo stored as RGB."""
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    return int(max(np.abs(b - g).max(), np.abs(g - r).max())) <= tol


def laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def luma_stats(img_bgr: np.ndarray) -> tuple[float, float, float, float]:
    """Mean luma, RMS contrast (std), fraction of clipped-dark (<=5) and clipped-bright (>=250) pixels."""
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(y.mean()), float(y.std()), float((y <= 5).mean()), float((y >= 250).mean())


_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def jpeg_quality_estimate(data: bytes) -> float | None:
    """Estimate IJG quality from the luminance quantization table (None if not a JPEG).

    Inverts the libjpeg scaling (scale = 5000/Q for Q<50, 200-2Q otherwise). The estimate is
    an approximation: it is exact only for encoders using the standard IJG tables.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            q = getattr(im, "quantization", None)
            if not q or 0 not in q:
                return None
            table = np.array(q[0], dtype=np.float64)
    except Exception:
        return None
    # PIL returns the table in zigzag order; compare as a multiset via sorted ratio mean
    scale = float(np.mean(np.sort(table) / np.sort(_STD_LUMA_Q)) * 100.0)
    if scale <= 0:
        return None
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return round(float(np.clip(quality, 1.0, 100.0)), 1)


def phash64(img_bgr: np.ndarray) -> int:
    """64-bit DCT perceptual hash (32x32 grayscale → top-left 8x8 DCT, median threshold)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    med = np.median(dct[1:])
    bits = dct > med
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_groups(hashes: Mapping[str, int], max_dist: int = 3) -> dict[str, str]:
    """Group ids whose 64-bit hashes are within `max_dist` bits.

    Exact via pigeonhole: split into (max_dist+1) chunks; any pair within max_dist shares
    at least one identical chunk, so only same-chunk candidates are compared.
    Returns {id: group_id} for ids that belong to a group of size >= 2.
    """
    n_chunks = max_dist + 1
    width = 64 // n_chunks
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ids = sorted(hashes)
    for c in range(n_chunks):
        shift = c * width
        mask = (1 << width) - 1 if c < n_chunks - 1 else (1 << (64 - shift)) - 1
        buckets: dict[int, list[str]] = {}
        for i in ids:
            buckets.setdefault((hashes[i] >> shift) & mask, []).append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 5000:  # giant buckets = degenerate (e.g. flat images); handled by exact check
                continue
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    a, b = members[a_i], members[b_i]
                    if hamming64(hashes[a], hashes[b]) <= max_dist:
                        parent.setdefault(a, a)
                        parent.setdefault(b, b)
                        union(a, b)
    groups: dict[str, list[str]] = {}
    for i in parent:
        groups.setdefault(find(i), []).append(i)
    out: dict[str, str] = {}
    for root, members in groups.items():
        if len(members) >= 2:
            for m in members:
                out[m] = f"nd-{root}"
    return out


def iter_chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
