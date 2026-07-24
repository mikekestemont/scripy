"""Tests for the flat index: cosine search + same-fragment metric (synthetic)."""

from __future__ import annotations

import numpy as np

from scripy.index import FlatIndex


def _toy():
    # two fragments, two pages each; pages of a fragment point in a shared direction
    vecs = np.array([
        [1.0, 0.0, 0.0],   # F-a page 0
        [0.9, 0.1, 0.0],   # F-a page 1  (close to page 0)
        [0.0, 1.0, 0.0],   # F-b page 0
        [0.0, 0.9, 0.1],   # F-b page 1  (close to F-b page 0)
    ], dtype=np.float32)
    files = ["F-a__00.png", "F-a__01.png", "F-b__00.png", "F-b__01.png"]
    return FlatIndex(vecs, files, provenance={})


def test_fragment_id_from_filename_prefix():
    idx = _toy()
    assert idx.fragments == ["F-a", "F-a", "F-b", "F-b"]


def test_search_returns_sibling_first():
    idx = _toy()
    hits = idx.search(idx.index_of("F-a__00.png"), k=3)
    assert hits[0].filename == "F-a__01.png"
    assert hits[0].fragment_id == "F-a"
    assert hits[0].score > hits[1].score  # sorted descending


def test_same_fragment_eval_perfect_on_separable_toy():
    r = _toy().same_fragment_eval()
    assert r["pages"] == 4 and r["fragments"] == 2
    assert r["queries_with_positive"] == 4
    assert r["top1"] == 1.0
    assert r["mAP"] == 1.0
