#!/usr/bin/env bash
# Reproduce the "Ferguut scribe" working example end-to-end.
#
# F-eadz is a Middle Dutch Alexander compilation copied by the Ferguut scribe
# (Brabant, s. XIV med.). This script harvests a small Middle Dutch probe corpus,
# runs the exact mole preprocessing, embeds every page into the frozen universal
# VLAD space, and asks: what are the nearest hands to F-eadz fol. 1r?
#
# Prereqs (one-time):
#   pip install -e .            # scripy
#   pip install -e ../mole      # the encoder
#   models/pooled.pth           # the pinned pooled checkpoint (417 MB)
#   models/universal.codebook.npy   # the frozen universal K=100 codebook
# (both model files are downloaded from the GPU server; see README.)
#
# Usage:  bash scripts/demo_ferguut.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=models/pooled.pth
CODEBOOK=models/universal.codebook.npy
WORK=data/harvest/ferguut-demo
BIN=${WORK}-bin
NPY=data/harvest/ferguut-demo.npy
PROV=${WORK}/provenance.csv

for f in "$CKPT" "$CODEBOOK"; do
  [ -f "$f" ] || { echo "missing $f — see the Prereqs in this script"; exit 1; }
done

echo "== 1/4  harvest (Middle Dutch probe corpus, incl. F-eadz) =="
scripy harvest --seed data/seeds/middle-dutch.txt --out "$WORK" --pages-per-fragment 3

echo "== 2/4  prep: Sauvola binarization (match the codebook's regime) =="
mole prep "$WORK" --binarize sauvola --max-side 2048 --binarize-out "$BIN"

echo "== 3/4  embed into the frozen universal VLAD space =="
mole embed "$CKPT" "$BIN" "$NPY" --codebook-from "$CODEBOOK"

echo "== 4/4  retrieval =="
scripy eval "$NPY" "$PROV"
echo
echo "-- nearest pages to F-eadz fol. 1r (page-level smoke test) --"
scripy search "$NPY" "$PROV" --query F-eadz__00.png -k 8
echo
echo "Expected: F-eadz__01.png (fol. 1v, same scribe) ranks near the top."
echo
echo "-- nearest MANUSCRIPTS to F-eadz (object-level: mean over all its leaves) --"
scripy search "$NPY" "$PROV" --query F-eadz --by-object -k 8
echo
echo "This is the scholarly query: one hit per manuscript, ranked by the whole"
echo "object's averaged hand. On the full Middle Dutch corpus, F-iruj surfaces as"
echo "the #1 same-hand lead for the Ferguut scribe."
