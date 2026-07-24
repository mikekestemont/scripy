"""Zone-aware high-resolution crops via the IIIF Image API.

The harvest downloads a modest-resolution page (fast, small). `mole prep` then
detects the text zone on it. This module maps that zone to an Image API ``pct:``
region and re-fetches **just the text** at native resolution — which removes the
photographic backdrop, colour bars and rulers (a real retrieval confound: fragments
photographed with the same ruler look artificially similar) *and* raises the text
resolution, while downloading less than a full high-res page. This is the zone-aware
streaming fetch the blueprint envisions (BLUEPRINT §4, §10).

Input is a harvest dir that already has ``zones.json`` (from ``mole prep``) and
``provenance.csv`` (from ``scripy harvest``). Crops are written with the same
filenames, so downstream provenance joins are unchanged.
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from pathlib import Path

__all__ = ["fetch_region_crops"]

_UA = "scripy/0.0.1 (+https://github.com/mikekestemont/scripy; handwriting index)"
_FULL_RE = re.compile(r"/full/[^/]+/0/default\.jpg$")


def _service_base(image_url: str) -> str:
    """`.../fol_1r.jp2/full/1400,/0/default.jpg` -> `.../fol_1r.jp2`."""
    return _FULL_RE.sub("", image_url)


def _pct_region_url(service: str, bbox, img_size, *, size: str, pad_pct: float) -> str:
    x0, y0, x1, y1 = bbox
    w, h = img_size
    X = 100.0 * x0 / w - pad_pct
    Y = 100.0 * y0 / h - pad_pct
    W = 100.0 * (x1 - x0) / w + 2 * pad_pct
    H = 100.0 * (y1 - y0) / h + 2 * pad_pct
    X, Y = max(0.0, X), max(0.0, Y)
    W, H = min(100.0 - X, W), min(100.0 - Y, H)
    return f"{service}/pct:{X:.3f},{Y:.3f},{W:.3f},{H:.3f}/{size}/0/default.jpg"


def fetch_region_crops(
    harvest_dir: str | Path,
    out_dir: str | Path,
    *,
    size: str = "full",
    pad_pct: float = 1.0,
    delay: float = 0.3,
    log=print,
) -> Path:
    """Re-fetch each page's text zone at native resolution into ``out_dir``.

    Pages whose detection fell back to the whole image (no text found) are fetched
    whole at ``size`` — no worse than before. Returns ``out_dir``.
    """
    harvest_dir, out_dir = Path(harvest_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zones = json.loads((harvest_dir / "zones.json").read_text()).get("images", {})
    url_of: dict[str, str] = {}
    with (harvest_dir / "provenance.csv").open() as fh:
        for row in csv.DictReader(fh):
            url_of[row["filename"]] = row["image_url"]

    n, n_fallback = 0, 0
    for i, (fname, z) in enumerate(zones.items(), 1):
        src = url_of.get(fname)
        if not src:
            continue
        service = _service_base(src)
        bbox, img_size = z.get("bbox"), z.get("size")
        if z.get("fell_back") or not bbox:
            url = f"{service}/full/{size if size != 'full' else '1600,'}/0/default.jpg"
            n_fallback += 1
        else:
            url = _pct_region_url(service, bbox, img_size, size=size, pad_pct=pad_pct)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            (out_dir / fname).write_bytes(urllib.request.urlopen(req, timeout=90).read())  # noqa: S310
            n += 1
        except Exception as exc:  # noqa: BLE001 - one bad crop must not stop the batch
            log(f"  {fname}: region fetch FAILED ({exc})")
        if delay:
            time.sleep(delay)
        if i % 50 == 0:
            log(f"  {i}/{len(zones)} regions fetched")
    log(f"fetched {n} region crops ({n_fallback} whole-page fallbacks) -> {out_dir}")
    return out_dir
