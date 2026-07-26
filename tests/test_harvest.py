"""Tests for source resolution + the folio-label filter (no network)."""

from __future__ import annotations

from scripy.harvest import _FOLIO_LABEL, resolve_source


def test_resolve_bare_fragmentarium_id():
    s = resolve_source("F-eadz")
    assert s.id == "F-eadz"
    assert s.manifest_url.endswith("/F-eadz/manifest.json")
    assert s.overview_url == "https://fragmentarium.ms/overview/F-eadz"


def test_resolve_explicit_id_and_manifest_url():
    url = "https://digitalcollections.universiteitleiden.nl/iiif_manifest/item:1598537/manifest"
    s = resolve_source(f"L-ltk191 {url}")
    assert s.id == "L-ltk191"
    assert s.manifest_url == url
    assert s.overview_url == url


def test_resolve_bare_manifest_url_derives_leiden_item_id():
    url = "https://digitalcollections.universiteitleiden.nl/iiif_manifest/item:1598537/manifest"
    s = resolve_source(url)
    assert s.id == "item1598537"
    assert s.manifest_url == url


def test_folio_label_matches_folios_not_binding():
    keep = ["f001r", "f001v", "fol_12v", "f. 3r", "7v", "123R"]
    drop = ["band1", "opening3 (incl. f001r)", "front cover", "spine", ""]
    assert all(_FOLIO_LABEL.match(x) for x in keep)
    assert not any(_FOLIO_LABEL.match(x) for x in drop)
