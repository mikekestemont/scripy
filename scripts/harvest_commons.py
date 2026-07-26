#!/usr/bin/env python3
"""Pull a Wikimedia Commons category of manuscript page scans into a scripy harvest dir.

Some key Middle Dutch codices are on Wikimedia Commons as public-domain full scans but
are NOT served over IIIF — e.g. the Lancelotcompilatie (The Hague, KB, 129 A 10), 482
files, folios 001r-238v. scripy's `harvest` command is IIIF-only, so this is a separate,
deliberately standalone downloader. It writes the SAME `provenance.csv` columns as
`scripy harvest`, so the images feed the identical `mole prep --binarize sauvola →
mole embed → FlatIndex` pipeline afterwards.

Files are pulled via the stable `Special:FilePath/<file>` redirect. By default it fetches
the ORIGINAL upload (fast: no server-side thumbnailing; these masters are already
~1920 px wide); pass --width N to request a scaled JPEG instead. Downloads run
concurrently (--workers). Pages are ordered by folio (001r, 001v, 002r, …); non-folio
plates (front, iv, binding) sort last unless --folios-only drops them.

Usage (defaults target the Lancelotcompilatie):
    python scripts/harvest_commons.py
    python scripts/harvest_commons.py --category "Lancelotcompilatie KB 129A10" \
        --id KB129A10 --out data/harvest/lancelot --folios-only
    python scripts/harvest_commons.py --limit 4          # quick validation sample

stdlib only (urllib/json/csv/re + concurrent.futures), matching src/scripy/harvest.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_UA = "scripy/0.0.1 (+https://github.com/mikekestemont/scripy; handwriting index)"
_API = "https://commons.wikimedia.org/w/api.php"
_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
_FILEPAGE = "https://commons.wikimedia.org/wiki/"

#: Trailing page token in a Commons title, e.g. "… - KB 129 A 10 - 001r.jpg" -> "001r".
_TOKEN = re.compile(r"-\s*([^-]+?)\s*\.jpe?g$", re.IGNORECASE)
_FOLIO = re.compile(r"^(\d{1,4})([rv])$", re.IGNORECASE)


def _get(url: str, *, timeout: float = 120.0, retries: int = 6) -> bytes:
    """GET with polite backoff on Wikimedia rate-limiting (429) / unavailability (503).

    Honours ``Retry-After`` when present, else exponential backoff. BLUEPRINT §1.
    """
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < retries:
                ra = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(ra) if (ra and ra.isdigit()) else min(30.0, 2.0 ** attempt)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def list_category_files(category: str) -> list[str]:
    """Return all `File:…` titles in a Commons category (paginated)."""
    titles: list[str] = []
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmtype": "file",
        "cmlimit": "500", "format": "json",
    }
    while True:
        data = json.loads(_get(f"{_API}?{urllib.parse.urlencode(params)}"))
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
    return titles


def _sort_key(title: str) -> tuple:
    """Order by folio number then recto/verso; non-folio plates sort last."""
    m = _TOKEN.search(title)
    token = (m.group(1) if m else title).lower()
    fm = _FOLIO.match(token)
    if fm:
        return (0, int(fm.group(1)), 0 if fm.group(2) == "r" else 1, token)
    return (1, 0, 0, token)  # front matter / binding after the numbered folios


def _is_folio(title: str) -> bool:
    m = _TOKEN.search(title)
    return bool(m and _FOLIO.match(m.group(1).strip()))


def harvest_commons(category: str, out_dir: Path, *, source_id: str, width: int,
                    folios_only: bool, limit: int | None, workers: int, delay: float) -> Path:
    titles = list_category_files(category)
    if folios_only:
        titles = [t for t in titles if _is_folio(t)]
    titles.sort(key=_sort_key)
    if limit:
        titles = titles[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_path = out_dir / "provenance.csv"
    overview = f"{_FILEPAGE}Category:{urllib.parse.quote(category.replace(' ', '_'))}"

    # Build tasks: (index, commons filename, download url, local filename).
    tasks = []
    for i, title in enumerate(titles):
        fname_commons = title.split(":", 1)[1] if ":" in title else title
        url = f"{_FILEPATH}{urllib.parse.quote(fname_commons)}"
        if width:
            url += f"?width={width}"
        tasks.append((i, fname_commons, url, f"{source_id}__{i:03d}.jpg"))

    def _work(task):
        i, fname_commons, url, local = task
        data = _get(url)
        (out_dir / local).write_bytes(data)
        if delay:
            time.sleep(delay)  # per-worker throttle to stay under Wikimedia's rate limit
        return i, fname_commons, url, local, len(data)

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                i, fname_commons, url, local, nbytes = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                print(f"    {task[3]}: download FAILED ({task[1]}) — {exc}", flush=True)
                continue
            rows.append([local, source_id, i, f"{_FILEPAGE}{urllib.parse.quote(fname_commons)}",
                         url, overview, nbytes])
            done += 1
            print(f"  [{done}/{len(tasks)}] {local}", flush=True)

    rows.sort(key=lambda r: r[2])  # write provenance in folio (canvas_index) order
    with prov_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fragment_id", "canvas_index", "canvas_id",
                    "image_url", "overview_url", "bytes"])
        w.writerows(rows)
    print(f"harvested {done}/{len(tasks)} page(s) from Commons category {category!r} -> {out_dir}")
    return prov_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--category", default="Lancelotcompilatie KB 129A10",
                    help="Wikimedia Commons category name (without the 'Category:' prefix).")
    ap.add_argument("--id", dest="source_id", default="KB129A10",
                    help="Short id / filename prefix + same-manuscript grouping key.")
    ap.add_argument("--out", type=Path, default=Path("data/harvest/lancelot"),
                    help="Output image directory.")
    ap.add_argument("--width", type=int, default=0,
                    help="Scaled JPEG width via Special:FilePath (0 = original, fastest).")
    ap.add_argument("--folios-only", action="store_true",
                    help="Keep only folio-labelled pages (drop front matter/binding).")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of pages (0 = all).")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent downloads (keep low; Wikimedia 429s parallel bots).")
    ap.add_argument("--delay", type=float, default=0.25,
                    help="Per-worker seconds between downloads (politeness throttle).")
    args = ap.parse_args()
    harvest_commons(args.category, args.out, source_id=args.source_id, width=args.width,
                    folios_only=args.folios_only, limit=args.limit or None,
                    workers=args.workers, delay=args.delay)


if __name__ == "__main__":
    main()
