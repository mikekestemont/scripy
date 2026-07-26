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
import re
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

from scripy import iiif

__all__ = ["harvest", "read_seed_list", "resolve_source", "Source"]

_UA = "scripy/0.0.1 (+https://github.com/mikekestemont/scripy; handwriting index)"

#: A folio label like ``f001r``, ``fol_12v``, ``f. 3r`` or bare ``7v`` — used by the
#: opt-in ``folios_only`` filter to drop binding/flyleaf/opening canvases that many
#: full-codex manifests carry before the text pages.
_FOLIO_LABEL = re.compile(r"(f(ol)?\.?[_\s]*)?\d{1,4}[rv]$", re.IGNORECASE)


class Source(NamedTuple):
    """A resolved harvest source: a stable short ``id`` + its IIIF ``manifest_url``."""

    id: str
    manifest_url: str
    overview_url: str


def _looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _derive_id(url: str) -> str:
    """Best-effort short id from a bare manifest URL (e.g. Leiden ``item:1598537``)."""
    m = re.search(r"item[:/](\d+)", url)
    if m:
        return f"item{m.group(1)}"
    slug = re.sub(r"/(manifest|manifest\.json)$", "", url.rstrip("/")).rstrip("/")
    return re.sub(r"[^0-9A-Za-z]+", "-", slug.rsplit("/", 1)[-1])[:40] or "manifest"


def resolve_source(entry: str) -> Source:
    """Resolve a seed entry to a :class:`Source`.

    Accepts three forms, so one seed list can mix Fragmentarium and any other IIIF
    repository:

    * ``F-eadz`` — a bare Fragmentarium ID → its Fragmentarium manifest + overview.
    * ``<id> <manifest-url>`` — an explicit short id and a full manifest URL
      (recommended for non-Fragmentarium sources; the id becomes the filename prefix
      and the same-manuscript grouping key).
    * ``https://…/manifest`` — a bare manifest URL; the id is derived from it.
    """
    parts = entry.split(None, 1)
    if len(parts) == 2 and _looks_like_url(parts[1]):
        return Source(parts[0], parts[1].strip(), parts[1].strip())
    token = parts[0]
    if _looks_like_url(token):
        return Source(_derive_id(token), token, token)
    return Source(token, iiif.fragmentarium_manifest_url(token),
                  f"https://fragmentarium.ms/overview/{token}")


def read_seed_list(path: str | Path) -> list[str]:
    """Read a seed file: one entry per line, ``#`` comments and blanks ignored.

    Each line is a :func:`resolve_source` entry — a Fragmentarium ID, an
    ``<id> <manifest-url>`` pair, or a bare manifest URL.
    """
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
    sources: list[str],
    out_dir: str | Path,
    *,
    size: str = "1600,",
    pages_per_fragment: int | None = 4,
    folios_only: bool = False,
    delay: float = 0.3,
    log=print,
) -> Path:
    """Download pages for each source into ``out_dir``; write ``provenance.csv``.

    ``sources`` is a list of :func:`resolve_source` entries (Fragmentarium IDs,
    ``<id> <manifest-url>`` pairs, or bare manifest URLs) — so a Fragmentarium crawl
    and a full-codex harvest from any other IIIF repository share one code path.
    ``pages_per_fragment`` caps pages per source (None = all). ``folios_only`` keeps
    only canvases whose label looks like a folio (``f001r``, ``12v``…), dropping the
    binding/flyleaf/opening canvases that full-codex manifests carry up front.
    ``delay`` throttles requests to stay polite to the IIIF servers (BLUEPRINT §1).
    Returns the path to the provenance CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_path = out_dir / "provenance.csv"

    n_pages = 0
    with prov_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fragment_id", "canvas_index", "canvas_id",
                    "image_url", "overview_url", "bytes"])
        for fi, entry in enumerate(sources, 1):
            src = resolve_source(entry)
            try:
                man = iiif.load_manifest(src.manifest_url, manifest_id=src.id)
            except Exception as exc:  # noqa: BLE001 - one bad manifest must not kill the crawl
                log(f"[{fi}/{len(sources)}] {src.id}: manifest FAILED ({exc})")
                continue
            canvases = man.canvases
            if folios_only:
                canvases = [cv for cv in canvases if _FOLIO_LABEL.match((cv.label or "").strip())]
            if pages_per_fragment:
                canvases = canvases[:pages_per_fragment]
            got = 0
            for ci, cv in enumerate(canvases):
                url = cv.image_request(size)
                if not url:
                    continue
                fname = f"{src.id}__{ci:02d}.jpg"
                try:
                    nbytes = _download(url, out_dir / fname)
                except Exception as exc:  # noqa: BLE001
                    log(f"    {fname}: download FAILED ({exc})")
                    continue
                w.writerow([fname, src.id, ci, cv.id, url, src.overview_url, nbytes])
                got += 1
                n_pages += 1
                if delay:
                    time.sleep(delay)
            log(f"[{fi}/{len(sources)}] {src.id}: {got} page(s)  ·  {man.label[:48]}")
    log(f"harvested {n_pages} pages from {len(sources)} source(s) -> {out_dir}")
    return prov_path
