#!/usr/bin/env python3
"""Build a page-level scribe ground truth for the Lancelotcompilatie (KB 129 A 10).

Labels each harvested folio side with its copyist (A-E), so mole can be tested on a
*single, uniformly imaged codex* — no ruler/backdrop confound, no cross-collection
imaging differences: a clean writer-identification benchmark.

SOURCE OF TRUTH — the scribe segmentation is scraped from the author's own TEI data,
`mikekestemont/spl2018scribes` (`data/compilation/01Lanc.xml`, `02Perc.xml`), where
scribe changes are `<scribe id="X"/>` milestones tracked BY VERSE. The verse stints
(Lanceloet, 36947 verses = fol. 1-99) are:
    A 1-17475 · B 17476-18415 · C 18416-25536 (minor B corrections) ·
    D 25537-32175 · B 32176-32334 · E 32335-36598 · B 36599-36947
Perchevael (fol. 100-115) is all B; by consensus B (Velthem) copied the rest to 238.

⚠️ APPROXIMATION: the XML has NO folio markers, so verse->folio is a LINEAR
interpolation over the Lanceloet (~373 verses/folio). Scribe labels are authoritative;
folio boundaries are approximate (±1-2 folios). Folios straddling a scribe change are
marked mixed and EXCLUDED. (This corrects a provisional LLM scheme that wrongly treated
C as a minor corrector; C is in fact a major scribe, ~fol. 50-68.)

Usage:
    python scripts/make_lancelot_labels.py
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.parse
from collections import Counter
from pathlib import Path

SOURCE = "spl2018scribes-xml/verse-interp"

# Authoritative verse stints from the spl2018scribes TEI (Lanceloet, fol. 1-99).
_STINTS = [(1, 17475, "A"), (17476, 18415, "B"), (18416, 23150, "C"),
           (23151, 23170, "B"), (23171, 23209, "C"), (23210, 23230, "B"),
           (23231, 25536, "C"), (25537, 32175, "D"), (32176, 32334, "B"),
           (32335, 36598, "E"), (36599, 36947, "B")]
_LANC_VERSES, _LANC_FOLIOS = 36947, 99
_VPF = _LANC_VERSES / _LANC_FOLIOS  # ~373 verses per folio
_LAST_FOLIO = 238                   # last harvested folio (KB129A10 runs to 238v)


def folio_scribe(folio: int) -> tuple[str | None, bool]:
    """Return (dominant scribe, is_mixed) for a folio. None scribe => not labelled."""
    if folio > _LANC_FOLIOS:
        return "B", False                       # fol. 100-238: Perchevael + rest = B
    lo, hi = int((folio - 1) * _VPF) + 1, int(folio * _VPF)
    cov: Counter[str] = Counter()
    for s, e, sc in _STINTS:
        ov = max(0, min(hi, e) - max(lo, s) + 1)
        if ov:
            cov[sc] += ov
    if not cov:
        return None, False
    return cov.most_common(1)[0][0], len(cov) > 1


_TOKEN = re.compile(r"-\s*(\d{1,3})([rv])\.jpg$", re.IGNORECASE)


def build(provenance: Path, out: Path, *, drop_mixed: bool, mole_out: Path | None) -> None:
    rows = list(csv.DictReader(provenance.open()))
    out.parent.mkdir(parents=True, exist_ok=True)
    labelled, excluded, allpages = [], [], []
    for r in rows:
        m = _TOKEN.search(urllib.parse.unquote(r["canvas_id"]))
        if not m:
            continue
        folio, side = int(m.group(1)), m.group(2).lower()
        scribe, mixed = folio_scribe(folio)
        clean = scribe is not None and not (mixed and drop_mixed)
        allpages.append((r["filename"], scribe if clean else None))
        if clean:
            labelled.append((r["filename"], folio, side, scribe, mixed))
        else:
            excluded.append((folio, side))

    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "folio", "side", "scribe", "mixed", "source"])
        for fname, folio, side, scribe, mixed in labelled:
            w.writerow([fname, f"{folio:03d}", side, scribe, int(mixed), SOURCE])

    if mole_out is not None:
        # mole eval/viz schema: filename (binarized .png) -> hand_id; transition folios
        # kept as gallery distractors with an empty hand_id (unlabeled).
        mole_out.parent.mkdir(parents=True, exist_ok=True)
        with mole_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "hand_id"])
            for fname, scribe in allpages:
                w.writerow([Path(fname).with_suffix(".png").name, scribe or ""])
        print(f"wrote {mole_out}  (mole eval/viz labels; {sum(s is not None for _,s in allpages)} labelled)")

    dist = Counter(s for _, _, _, s, _ in labelled)
    print(f"wrote {out}  ({len(labelled)} labelled, {len(excluded)} excluded)")
    for scribe in ("A", "B", "C", "D", "E"):
        print(f"  {scribe}: {dist[scribe]:3d} page-sides")
    mixedf = sorted({f for f, _ in excluded})
    print(f"  excluded transition folios: {mixedf}")
    print("  NOTE: scribe labels authoritative (spl2018scribes); folio boundaries "
          "interpolated (±1-2 folios).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--provenance", type=Path,
                    default=Path("data/harvest/lancelot/provenance.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/labels/lancelot-scribes.csv"))
    ap.add_argument("--keep-mixed", action="store_true",
                    help="Keep transition folios (labelled by dominant scribe, mixed=1).")
    ap.add_argument("--mole-out", type=Path, default=None,
                    help="Also write a mole eval/viz labels.csv here (e.g. data/harvest/lancelot-bin/labels.csv).")
    args = ap.parse_args()
    build(args.provenance, args.out, drop_mixed=not args.keep_mixed, mole_out=args.mole_out)


if __name__ == "__main__":
    main()
