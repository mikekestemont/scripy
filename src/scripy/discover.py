"""Discovery: enumerate fragments from a repository (BLUEPRINT §2, §3 Stage 3).

Fragmentarium has no documented public enumeration API, but its faceted search is
server-rendered and stable: a language-code facet returns the matching fragments'
``/overview/F-xxxx`` links. Middle Dutch is ISO 639-2 ``dum``. This is the pluggable
``Discoverer`` the blueprint calls for — each repository gets its own; here is
Fragmentarium's.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request

__all__ = ["fragmentarium_by_language", "MIDDLE_DUTCH"]

MIDDLE_DUTCH = "dum"  # ISO 639-2 language code for Middle Dutch (ca. 1050-1550)

_SEARCH = "https://fragmentarium.ms/search/"
_UA = "scripy/0.0.1 (+https://github.com/mikekestemont/scripy; handwriting index)"
_ID_RE = re.compile(r"/overview/(F-[a-z0-9]{4})")


def fragmentarium_by_language(lang_code: str = MIDDLE_DUTCH, *, timeout: float = 60.0) -> list[str]:
    """Return the sorted Fragmentarium fragment IDs whose *text* language is ``lang_code``.

    Uses the ``text_lang_code_facet`` search facet, e.g. ``dum`` for Middle Dutch.
    """
    qs = urllib.parse.urlencode({f"aSelectedFacets[text_lang_code_facet][]": lang_code})
    req = urllib.request.Request(f"{_SEARCH}?{qs}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        html = resp.read().decode("utf-8", "replace")
    return sorted(set(_ID_RE.findall(html)))
