"""The mole bridge (BLUEPRINT §7).

scripy owns no model code. This module streams IIIF page images to a temporary
directory, hands the directory to ``mole``'s high-level ``embed()`` (which loads
the pinned checkpoint, extracts foreground patch-window descriptors, and applies a
**frozen** VLAD codebook when ``codebook`` is given), reads the resulting vectors
back, and deletes the images. No pixels are persisted beyond the temp dir.

``mole`` (and therefore ``torch``) is imported lazily so the pure-IIIF Stage-0 path
in :mod:`scripy.iiif` works without the model stack installed.
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

from scripy.iiif import Canvas, Manifest

_USER_AGENT = "scripy/0.0.1 (+https://github.com/; handwriting index)"


def _safe_name(canvas_id: str, index: int) -> str:
    """A filesystem-safe, order-preserving stem so vectors map back to canvases."""
    tail = canvas_id.rstrip("/").split("/")[-1] or f"canvas{index}"
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in tail)
    return f"{index:04d}_{keep}"


def _download(url: str, dest: Path, *, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted IIIF host)
        dest.write_bytes(resp.read())


def embed_manifest(
    manifest: Manifest,
    *,
    checkpoint: str | Path,
    codebook: str | Path | None = None,
    size: str = "1024,",
    tmp_root: str | Path | None = None,
) -> Iterator[tuple[Canvas, np.ndarray]]:
    """Yield ``(canvas, vector)`` for each page, deleting each image after encode.

    Uses ``mole.embed.extract.embed`` with ``pooling='vlad'`` and, when a frozen
    ``codebook`` is supplied, ``codebook_from=`` so every page lands in the one
    shared embedding space (BLUEPRINT §8). The per-canvas filename stem carries the
    mapping between mole's output rows and the source canvases.
    """
    from mole.embed.extract import embed as mole_embed  # lazy: pulls in torch

    stem_to_canvas: dict[str, Canvas] = {}
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix="scripy_") as tmp:
        tmp = Path(tmp)
        img_dir = tmp / "pages"
        img_dir.mkdir()

        for i, canvas in enumerate(manifest.canvases):
            url = canvas.image_request(size)
            if not url:
                continue
            stem = _safe_name(canvas.id, i)
            _download(url, img_dir / f"{stem}.jpg")
            stem_to_canvas[stem] = canvas

        if not stem_to_canvas:
            return

        out = tmp / "vectors.npy"
        mole_embed(
            checkpoint=checkpoint,
            input_dir=img_dir,
            output=out,
            pooling="vlad",
            codebook_from=codebook,
        )

        vectors = np.load(out)
        mapping = json.loads((tmp / "vectors.mapping.json").read_text())
        # mapping records the image path per row, in output order.
        paths = mapping.get("images") or mapping.get("paths") or []
        for row, image_path in enumerate(paths):
            stem = Path(image_path).stem
            canvas = stem_to_canvas.get(stem)
            if canvas is not None:
                yield canvas, vectors[row]
    # TemporaryDirectory removal deletes every streamed image (delete-after-index).
