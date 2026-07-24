"""Minimal IIIF Presentation-API client — the Stage-0 walking skeleton.

Parsing is a pure function over already-loaded JSON so it is trivially testable
offline against the checked-in fixtures. Only ``load_manifest`` touches the
network. Both Presentation API **v2** (Fragmentarium: ``sc:Manifest`` +
``sequences``/``canvases``) and **v3** (``Manifest`` + ``items``/
``AnnotationPage``) are normalised to the same :class:`Canvas` shape, so widening
the crawl to v3 repositories later is not a rewrite.

scripy owns none of the model. This module's only job is to answer, for a given
manifest: *what are the canvases, and at what URL do I fetch each page image?*
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

__all__ = ["Canvas", "Manifest", "parse_manifest", "load_manifest", "FRAGMENTARIUM_MANIFEST"]

FRAGMENTARIUM_MANIFEST = "https://fragmentarium.ms/metadata/iiif/{fragment_id}/manifest.json"

_USER_AGENT = "scripy/0.0.1 (+https://github.com/; handwriting index; contact repo owner)"


@dataclass(frozen=True)
class Canvas:
    """One page. ``image_service`` is the IIIF Image API base (no ``/full/...``)."""

    id: str
    label: str
    width: int | None
    height: int | None
    image_service: str | None
    #: A directly-usable image URL when the resource has no Image API service.
    image_url: str | None = None

    def image_request(self, size: str = "full") -> str | None:
        """Build an Image API request URL, e.g. ``{service}/full/1024,/0/default.jpg``.

        ``size`` follows the IIIF Image API size syntax: ``full``, ``max``,
        ``1024,`` (width 1024, aspect kept), ``,1024`` (height), ``!1024,1024``
        (fit within), ``pct:50``. Falls back to a bare ``image_url`` when the
        canvas exposes no Image API service.
        """
        if self.image_service:
            return f"{self.image_service}/full/{size}/0/default.jpg"
        return self.image_url


@dataclass(frozen=True)
class Manifest:
    id: str
    label: str
    language: str | None
    canvases: list[Canvas]
    presentation_version: int  # 2 or 3

    def __len__(self) -> int:
        return len(self.canvases)


# --------------------------------------------------------------------------- #
# Parsing (pure; no network)                                                   #
# --------------------------------------------------------------------------- #

def _label(node: Any) -> str:
    """IIIF labels are a plain string (v2) or a language-map dict (v3)."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        # v3 language map: {"en": ["..."], "none": ["..."]}
        for key in ("none", "en", "@none"):
            if key in node and node[key]:
                return node[key][0]
        for values in node.values():
            if isinstance(values, list) and values:
                return values[0]
            if isinstance(values, str):
                return values
    if isinstance(node, list) and node:
        return _label(node[0])
    return ""


def _first_service_id(service: Any) -> str | None:
    """Pull the Image API service base id from the many shapes IIIF allows."""
    if service is None:
        return None
    if isinstance(service, list):
        for entry in service:
            found = _first_service_id(entry)
            if found:
                return found
        return None
    if isinstance(service, dict):
        sid = service.get("@id") or service.get("id")
        if sid:
            return sid.rstrip("/")
    return None


def _language(metadata: Any) -> str | None:
    """Best-effort language read from a v2 ``metadata`` list of label/value pairs."""
    if not isinstance(metadata, list):
        return None
    for pair in metadata:
        if not isinstance(pair, dict):
            continue
        if _label(pair.get("label")).strip().lower() in {"language", "text language"}:
            return _label(pair.get("value")) or None
    return None


def _parse_v2(data: dict, manifest_id: str) -> Manifest:
    canvases: list[Canvas] = []
    for seq in data.get("sequences", []):
        for cv in seq.get("canvases", []):
            image = (cv.get("images") or [{}])[0]
            resource = image.get("resource", {}) or {}
            service = _first_service_id(resource.get("service"))
            canvases.append(
                Canvas(
                    id=cv.get("@id", ""),
                    label=_label(cv.get("label")),
                    width=cv.get("width"),
                    height=cv.get("height"),
                    image_service=service,
                    image_url=None if service else resource.get("@id"),
                )
            )
    return Manifest(
        id=manifest_id,
        label=_label(data.get("label")),
        language=_language(data.get("metadata")),
        canvases=canvases,
        presentation_version=2,
    )


def _parse_v3(data: dict, manifest_id: str) -> Manifest:
    canvases: list[Canvas] = []
    for cv in data.get("items", []):
        service = None
        image_url = None
        for anno_page in cv.get("items", []):
            for anno in anno_page.get("items", []):
                body = anno.get("body", {}) or {}
                service = _first_service_id(body.get("service"))
                image_url = body.get("id")
                if service or image_url:
                    break
            if service or image_url:
                break
        canvases.append(
            Canvas(
                id=cv.get("id", ""),
                label=_label(cv.get("label")),
                width=cv.get("width"),
                height=cv.get("height"),
                image_service=service,
                image_url=None if service else image_url,
            )
        )
    return Manifest(
        id=manifest_id,
        label=_label(data.get("label")),
        language=None,  # v3 language lives in metadata language-maps; add when needed
        canvases=canvases,
        presentation_version=3,
    )


def parse_manifest(data: dict, manifest_id: str = "") -> Manifest:
    """Normalise a IIIF Presentation v2 or v3 manifest to :class:`Manifest`."""
    ctx = data.get("@context", "")
    ctx_str = " ".join(ctx) if isinstance(ctx, list) else str(ctx)
    type_ = data.get("@type") or data.get("type") or ""
    is_v2 = "presentation/2" in ctx_str or type_ == "sc:Manifest" or "sequences" in data
    manifest_id = manifest_id or data.get("@id") or data.get("id") or ""
    return _parse_v2(data, manifest_id) if is_v2 else _parse_v3(data, manifest_id)


# --------------------------------------------------------------------------- #
# Loading (network)                                                            #
# --------------------------------------------------------------------------- #

def load_manifest(url: str, *, timeout: float = 30.0, manifest_id: str = "") -> Manifest:
    """Fetch and parse a IIIF manifest. Read-only GET, descriptive User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted IIIF host)
        data = json.loads(resp.read().decode("utf-8"))
    return parse_manifest(data, manifest_id=manifest_id or url)


def fragmentarium_manifest_url(fragment_id: str) -> str:
    return FRAGMENTARIUM_MANIFEST.format(fragment_id=fragment_id)
