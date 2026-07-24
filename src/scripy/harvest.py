"""Harvest: stream fragment page images from IIIF into a local working dir.

This is the download stage of BLUEPRINT §4. It resolves each fragment ID to its
IIIF manifest, downloads a bounded number of page images at a chosen size, and
writes a ``provenance.csv`` so every harvested file can be traced back to its
fragment, canvas, and source viewer. Filenames are ``{fragment_id}__{NN}.jpg`` so
the fragment ID (a stand-in "hand" for same-manuscript evaluation) is recoverable
from the filename alone.

Stage 1 keeps images on disk for the prep+embed steps; the streaming/delete-after-
index behaviour is introduced in Stage 2.
"""

from __future__ import annotations

import csv
import time
import urllib.request
from pathlib import Path

from scripy import iiif

__all__ = ["harvest", "read_seed_list"]

_UA = "scripy/0.0.1 (+https://github.com/mikekestemont/scripy; handwriting index)"


def read_seed_list(path: str | Path) -> list[str]:
    """Read a seed file: one fragment ID per line, ``#`` comments and blanks ignored."""
    ids: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def _download(url: str, dest: Path, *, timeout: float = 90.0) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted IIIF host)
        data = resp.read()
    dest.write_bytes(data)
    return len(data)


def harvest(
    fragment_ids: list[str],
    out_dir: str | Path,
    *,
    size: str = "1600,",
    pages_per_fragment: int | None = 4,
    delay: float = 0.3,
    log=print,
) -> Path:
    """Download pages for each fragment into ``out_dir``; write ``provenance.csv``.

    ``pages_per_fragment`` caps pages per fragment (None = all). ``delay`` throttles
    requests to stay polite to the IIIF servers (BLUEPRINT §1). Returns the path to
    the provenance CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_path = out_dir / "provenance.csv"

    n_pages = 0
    with prov_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fragment_id", "canvas_index", "canvas_id",
                    "image_url", "overview_url", "bytes"])
        for fi, fid in enumerate(fragment_ids, 1):
            try:
                man = iiif.load_manifest(iiif.fragmentarium_manifest_url(fid), manifest_id=fid)
            except Exception as exc:  # noqa: BLE001 - one bad manifest must not kill the crawl
                log(f"[{fi}/{len(fragment_ids)}] {fid}: manifest FAILED ({exc})")
                continue
            canvases = man.canvases[:pages_per_fragment] if pages_per_fragment else man.canvases
            got = 0
            for ci, cv in enumerate(canvases):
                url = cv.image_request(size)
                if not url:
                    continue
                fname = f"{fid}__{ci:02d}.jpg"
                try:
                    nbytes = _download(url, out_dir / fname)
                except Exception as exc:  # noqa: BLE001
                    log(f"    {fname}: download FAILED ({exc})")
                    continue
                w.writerow([fname, fid, ci, cv.id, url,
                            f"https://fragmentarium.ms/overview/{fid}", nbytes])
                got += 1
                n_pages += 1
                if delay:
                    time.sleep(delay)
            log(f"[{fi}/{len(fragment_ids)}] {fid}: {got} page(s)  ·  {man.label[:48]}")
    log(f"harvested {n_pages} pages from {len(fragment_ids)} fragments -> {out_dir}")
    return prov_path
