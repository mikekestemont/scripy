"""scripy command-line interface.

Stage 0 is real and runnable today: ``scripy manifest <fragment_id>`` needs only
``requests``-free stdlib + typer. Commands that touch the model (``embed``) import
``mole`` lazily, so the IIIF path works even where the full mole/torch stack is not
installed. See ``docs/BLUEPRINT.md`` for the staged roadmap.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from scripy import iiif

app = typer.Typer(add_completion=False, help="Streaming handwriting index over IIIF.")


@app.command()
def manifest(
    fragment_id: str = typer.Argument(..., help="Fragmentarium ID (e.g. F-eadz) or manifest URL."),
    url: str = typer.Option("", "--url", help="Explicit manifest URL (overrides fragment_id)."),
    size: str = typer.Option("full", "--size", help="IIIF Image API size, e.g. full, 1024,, pct:50."),
    fixture: Path = typer.Option(None, "--fixture", help="Parse a local manifest JSON instead of fetching."),
) -> None:
    """List a manifest's canvases and their image-request URLs (Stage 0)."""
    if fixture is not None:
        man = iiif.parse_manifest(json.loads(fixture.read_text()), manifest_id=fragment_id)
    else:
        man_url = url or (fragment_id if fragment_id.startswith("http") else iiif.fragmentarium_manifest_url(fragment_id))
        man = iiif.load_manifest(man_url, manifest_id=fragment_id)

    typer.secho(f"{man.label}", bold=True)
    typer.echo(f"  id={man.id}  IIIF Presentation v{man.presentation_version}  language={man.language or '—'}")
    typer.echo(f"  {len(man)} canvas(es):")
    for i, cv in enumerate(man.canvases):
        typer.echo(f"    [{i}] {cv.label or cv.id}  ({cv.width}x{cv.height})")
        typer.echo(f"        {cv.image_request(size)}")


@app.command()
def embed(
    fragment_id: str = typer.Argument(..., help="Fragmentarium ID or manifest URL."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="mole checkpoint (the pinned pooled model)."),
    codebook: Path = typer.Option(None, "--codebook", help="Frozen universal VLAD codebook (.codebook.npy)."),
    size: str = typer.Option("1024,", "--size", help="IIIF size to fetch for encoding."),
) -> None:
    """Encode a manifest's pages into VLAD vectors via ``mole`` (Stage 0/1 bridge).

    Downloads each canvas image to a temp file, hands it to mole, prints vector
    shapes, and deletes the temp file. This is the thin bridge described in
    BLUEPRINT §7 — the model lives in mole; scripy only streams and cleans up.
    """
    try:
        from scripy.encode import embed_manifest  # lazy: imports mole/torch
    except ImportError as exc:  # pragma: no cover - depends on optional mole install
        raise typer.BadParameter(
            f"The embed command needs `mole` installed (pip install -e ../mole). Import failed: {exc}"
        ) from exc

    man_url = fragment_id if fragment_id.startswith("http") else iiif.fragmentarium_manifest_url(fragment_id)
    man = iiif.load_manifest(man_url, manifest_id=fragment_id)
    for cv, vec in embed_manifest(man, checkpoint=checkpoint, codebook=codebook, size=size):
        typer.echo(f"{cv.id}\t{tuple(vec.shape)}")


@app.command()
def discover(
    lang: str = typer.Option("dum", "--lang", help="ISO 639-2 text-language code (dum = Middle Dutch)."),
    out: Path = typer.Option(None, "--out", help="Write the fragment IDs to this file (one per line)."),
) -> None:
    """List Fragmentarium fragment IDs for a text language (Stage 3 discoverer)."""
    from scripy.discover import fragmentarium_by_language
    ids = fragmentarium_by_language(lang)
    typer.echo(f"{len(ids)} fragments with text language '{lang}'")
    if out:
        out.write_text("\n".join(ids) + "\n")
        typer.echo(f"wrote {out}")
    else:
        for fid in ids:
            typer.echo(fid)


@app.command()
def harvest(
    seed: Path = typer.Option("data/seeds/middle-dutch.txt", "--seed", help="Seed list of fragment IDs."),
    out: Path = typer.Option("data/harvest/middle-dutch", "--out", help="Output image directory."),
    size: str = typer.Option("1600,", "--size", help="IIIF Image API size for harvested pages."),
    pages_per_fragment: int = typer.Option(4, "--pages-per-fragment", help="Cap pages per fragment (0 = all)."),
) -> None:
    """Download page images for a seed list into a working dir with provenance."""
    from scripy.harvest import harvest as run_harvest, read_seed_list
    ids = read_seed_list(seed)
    typer.echo(f"harvesting {len(ids)} fragments -> {out}")
    run_harvest(ids, out, size=size, pages_per_fragment=pages_per_fragment or None)


@app.command()
def crop(
    harvest: Path = typer.Option(..., "--harvest", help="Harvest dir with zones.json + provenance.csv."),
    out: Path = typer.Option(..., "--out", help="Output dir for high-res text-region crops."),
    size: str = typer.Option("full", "--size", help="IIIF size for the region (full = native)."),
    per_column: bool = typer.Option(False, "--per-column", help="One crop per text column (vs the union box)."),
) -> None:
    """Re-fetch each page's detected text zone(s) at native resolution from IIIF."""
    from scripy.zones import fetch_region_crops
    fetch_region_crops(harvest, out, per_column=per_column, size=size)


@app.command()
def eval(
    npy: Path = typer.Argument(..., help="Embeddings .npy written by `mole embed`."),
    provenance: Path = typer.Argument(..., help="provenance.csv from `scripy harvest`."),
) -> None:
    """Label-free same-fragment retrieval sanity metric over the index."""
    from scripy.index import FlatIndex
    idx = FlatIndex.load(npy, provenance)
    r = idx.same_fragment_eval()
    typer.echo(f"pages={r['pages']}  fragments={r['fragments']}  "
               f"queries={r['queries_with_positive']}")
    typer.secho(f"same-fragment  Top-1={r['top1']:.3f}  mAP={r['mAP']:.3f}", bold=True)


@app.command()
def search(
    npy: Path = typer.Argument(..., help="Embeddings .npy written by `mole embed`."),
    provenance: Path = typer.Argument(..., help="provenance.csv from `scripy harvest`."),
    query: str = typer.Option(..., "--query", help="Query filename (e.g. F-eadz__00.jpg)."),
    k: int = typer.Option(5, "-k", help="Number of neighbours to return."),
) -> None:
    """Return the k nearest hands to a query page."""
    from scripy.index import FlatIndex
    idx = FlatIndex.load(npy, provenance)
    qi = idx.index_of(query)
    typer.secho(f"query: {query}  (fragment {idx.fragments[qi]})", bold=True)
    for h in idx.search(qi, k):
        flag = "  ← same fragment" if h.fragment_id == idx.fragments[qi] else ""
        typer.echo(f"  #{h.rank}  {h.score:.3f}  {h.filename:<16} {h.fragment_id}{flag}")


def main() -> None:  # console-script entry point
    app()


if __name__ == "__main__":
    main()
