"""scripy — a streaming handwriting search index over IIIF, powered by ``mole``.

See ``docs/BLUEPRINT.md`` for the design and staged roadmap. Stage 0 (the
walking skeleton) lives in :mod:`scripy.iiif`.
"""

from __future__ import annotations

__version__ = "0.0.1"

from scripy.iiif import Canvas, Manifest, load_manifest, parse_manifest  # noqa: E402

__all__ = ["Canvas", "Manifest", "load_manifest", "parse_manifest", "__version__"]
