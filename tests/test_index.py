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


def test_object_vectors_are_one_unit_vector_per_object():
    idx = _toy()
    ids, ovecs = idx.object_vectors()
    assert ids == ["F-a", "F-b"]
    assert ovecs.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(ovecs, axis=1), 1.0, atol=1e-6)
    # F-a's object vector is the renormalized mean of its two page unit-vectors,
    # so it points between them (positive first two comps, ~zero third).
    assert ovecs[0][0] > 0.9 and ovecs[0][2] < 1e-6


def test_search_object_ranks_sibling_object_first():
    idx = _toy()
    hits = idx.search_object("F-a", k=1)
    assert hits[0].fragment_id == "F-b"  # only other object
    assert hits[0].n_pages == 2


def test_resolve_object_accepts_id_and_filename():
    idx = _toy()
    assert idx._resolve_object("F-a") == "F-a"
    assert idx._resolve_object("F-a__01.png") == "F-a"


def test_object_split_eval_pairs_halves_on_separable_toy():
    r = _toy().object_split_eval()
    assert r["objects_with_2plus_pages"] == 2
    assert r["halves"] == 4
    assert r["top1"] == 1.0
