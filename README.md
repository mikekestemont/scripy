# scripy

**A streaming handwriting search index over IIIF, powered by the [`mole`](../mole) encoder.**

`scripy` turns the world's [IIIF](https://iiif.io) image collections into a
searchable index of *handwriting*: point it at a manuscript page and it returns
pages, across every harvested collection, written by a visually similar hand — a
direct tool for identifying scribes and **reassembling dispersed fragments**.

It does this **without building an image archive**. Pages are streamed from IIIF
servers, encoded into a fixed vector, and deleted. Only vectors and provenance are
kept.

- **`scripy` owns:** IIIF discovery + harvesting, streaming/bounded-memory
  orchestration, restartable state, the vector index, and search.
- **`mole` owns:** the trained encoder, patch-window descriptors, the frozen VLAD
  codebook, and the retrieval metrics. scripy imports it — no model code is
  duplicated here.

See **[docs/BLUEPRINT.md](docs/BLUEPRINT.md)** for the full design and the staged,
prototype-first roadmap.

## First case study: Fragmentarium's Middle Dutch fragments

We start small and in-domain: the Middle Dutch material in
[Fragmentarium](https://fragmentarium.ms) (dozens of fragments, IIIF-native, same
language/region `mole` was trained on). The prototype grows in complexity from
there — one manifest → a mini-corpus → a streaming crawl → a served search API.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # Stage 0: pure IIIF harvesting (typer + numpy)
pip install -e '.[embed]'   # + the mole encoder (also: pip install -e ../mole)
pip install -e '.[dev]'     # + pytest
```

## Try it (Stage 0 — the walking skeleton)

List the canvases of the reference fragment and their IIIF image URLs, fully
offline from the checked-in manifest:

```bash
scripy manifest F-eadz --fixture tests/fixtures/F-eadz.manifest.json
```

Or fetch it live from Fragmentarium:

```bash
scripy manifest F-eadz
```

Encode a fragment's pages into VLAD vectors via `mole` (needs `.[embed]` and a
pinned checkpoint + frozen codebook):

```bash
scripy embed F-eadz --checkpoint /path/to/pooled.pth --codebook /path/to/universal.codebook.npy
```

## Search Middle Dutch handwriting (Stage 1)

Needs the two model artifacts in `models/` (downloaded from the GPU server): the
pinned checkpoint `pooled.pth` and the frozen universal codebook
`universal.codebook.npy`. Then the whole pipeline is four commands:

```bash
scripy discover --lang dum --out data/seeds/middle-dutch-all.txt   # 314 fragments
scripy harvest  --seed data/seeds/middle-dutch.txt --out data/harvest/md
mole   prep     data/harvest/md --binarize sauvola --max-side 2048 --binarize-out data/harvest/md-bin
mole   embed    models/pooled.pth data/harvest/md-bin data/harvest/md.npy --codebook-from models/universal.codebook.npy
```

### The Ferguut-scribe working example

F-eadz is a Middle Dutch *Alexander* compilation in the hand of the **Ferguut
scribe**. Ask the index for the nearest hands to its first leaf:

```bash
scripy search data/harvest/md.npy data/harvest/md/provenance.csv --query F-eadz__00.png -k 8
```

`#1` should be `F-eadz__01.png` — fol. 1v, the same scribe. The label-free
same-fragment sanity metric (can each leaf find its siblings?):

```bash
scripy eval data/harvest/md.npy data/harvest/md/provenance.csv
```

**Reproduce it all in one go** (harvest → prep → embed → search):

```bash
bash scripts/demo_ferguut.sh
```

## Test

```bash
pytest
```

## Status

Stage 0 (IIIF client + mole bridge) is scaffolded and unit-tested offline. Stages
1–5 (mini-corpus index, streaming/restartable crawl, scale-out, serving, continual
maintenance) are specified in the blueprint and built in order.

## License

MIT © Mike Kestemont
