"""Zone-aware high-resolution crops via the IIIF Image API.

The harvest downloads a modest-resolution page (fast, small). `mole prep` then
detects text zones on it. This module maps a zone to an Image API ``pct:`` region
and re-fetches **just the text** at native resolution — which removes the
photographic backdrop, colour bars and rulers (a real retrieval confound: fragments
photographed with the same ruler look artificially similar) *and* raises the text
resolution, while downloading less than a full high-res page.

Two modes:

- **union** (default): one crop per page over the bounding box of all text zones.
- **per-column** (``per_column=True``): one crop per detected text *column*. On
  multi-column / multi-strip fragment guard-leaves the union box sweeps up the blank
  band between columns, inter-column gutters and library stamps, so most 224px
  windows land on emptiness; per-column crops give dense, upright, single-column
  text — a much crisper view of the script. Detections come from
  ``zones.json['images'][file]['detections']`` (``[family, conf, x0,y0,x1,y1]``);
  overlapping ``Text``/``Text_Main`` boxes for the same column are merged by NMS.

Input is a harvest dir with ``zones.json`` (from ``mole prep``) and ``provenance.csv``
(from ``scripy harvest``). Union crops keep the page filename; per-column crops append
``_cM`` so the fragment id (the ``{id}__NN`` prefix) is still recoverable.
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
_TEXT_FAMILIES = ("Text_Main", "Text")


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


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua else 0.0


def _column_boxes(detections, *, min_conf: float, iou_thr: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Deduplicate overlapping text detections into one box per column (greedy NMS)."""
    cand = [(conf, (x0, y0, x1, y1))
            for fam, conf, x0, y0, x1, y1 in detections
            if fam in _TEXT_FAMILIES and conf >= min_conf]
    cand.sort(key=lambda c: -c[0])
    kept: list[tuple[int, int, int, int]] = []
    for _, box in cand:
        if all(_iou(box, k) < iou_thr for k in kept):
            kept.append(box)
    kept.sort(key=lambda b: (b[0], b[1]))  # reading order: left-to-right, top-to-bottom
    return kept


def _crops_for_image(fname, z, service, *, per_column, size, pad_pct, min_conf):
    """Yield (out_filename, url) for one harvested page."""
    stem, ext = Path(fname).stem, Path(fname).suffix
    bbox, img_size = z.get("bbox"), z.get("size")
    if z.get("fell_back") or not bbox:
        whole = f"{service}/full/{size if size != 'full' else '1600,'}/0/default.jpg"
        yield fname, whole, True
        return
    if per_column:
        cols = _column_boxes(z.get("detections", []), min_conf=min_conf)
        if cols:
            for m, box in enumerate(cols):
                yield f"{stem}_c{m}{ext}", _pct_region_url(service, box, img_size, size=size, pad_pct=pad_pct), False
            return
        # no confident columns -> fall back to the union box
    yield fname, _pct_region_url(service, bbox, img_size, size=size, pad_pct=pad_pct), False


def fetch_region_crops(
    harvest_dir: str | Path,
    out_dir: str | Path,
    *,
    per_column: bool = False,
    size: str = "full",
    pad_pct: float = 1.0,
    min_conf: float = 0.5,
    delay: float = 0.1,
    log=print,
) -> Path:
    """Re-fetch text zones at native resolution into ``out_dir``. Returns ``out_dir``."""
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
        for out_name, url, is_fallback in _crops_for_image(
            fname, z, service, per_column=per_column, size=size, pad_pct=pad_pct, min_conf=min_conf
        ):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                (out_dir / out_name).write_bytes(urllib.request.urlopen(req, timeout=90).read())  # noqa: S310
                n += 1
                n_fallback += int(is_fallback)
            except Exception as exc:  # noqa: BLE001 - one bad crop must not stop the batch
                log(f"  {out_name}: region fetch FAILED ({exc})")
            if delay:
                time.sleep(delay)
        if i % 100 == 0:
            log(f"  {i}/{len(zones)} pages -> {n} crops")
    log(f"fetched {n} crops ({n_fallback} whole-page fallbacks) from {len(zones)} pages -> {out_dir}")
    return out_dir
