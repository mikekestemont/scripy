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


def main() -> None:  # console-script entry point
    app()


if __name__ == "__main__":
    main()
