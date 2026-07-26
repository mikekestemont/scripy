"""Flat vector index + search (BLUEPRINT §3 Stage 1; FAISS replaces this at Stage 3).

Loads the page vectors mole wrote (``.npy`` + ``.mapping.json``), joins them to the
harvest ``provenance.csv``, and answers cosine nearest-neighbour queries. Also
provides a label-free sanity metric: *same-fragment retrieval* — can a page find the
other leaves of its own fragment? With Fragmentarium's dispersed fragments this is a
lower bound on same-hand retrieval (leaves of one manuscript are often catalogued as
separate fragments), but it needs no hand labels and exercises the whole pipeline.

On top of page-level search, the index also pools pages into one vector per *object*
(a manuscript / ``fragment_id``) by averaging its page unit-vectors — see
:meth:`FlatIndex.object_vectors`. Object→object search is the query a scholar actually
wants ("which other books are in this hand?"); averaging denoises per-leaf variation
and returns one ranked hit per manuscript. We ignore, for now, that one document may
contain more than one hand.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FlatIndex", "Hit", "ObjectHit"]


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    filename: str
    fragment_id: str
    overview_url: str


@dataclass(frozen=True)
class ObjectHit:
    rank: int
    score: float
    fragment_id: str
    n_pages: int
    overview_url: str


def _mapping_paths(mapping: dict) -> list[str]:
    # mole writes {"rows": [{"row": i, "image": path}, ...]} in output order.
    if isinstance(mapping.get("rows"), list):
        rows = sorted(mapping["rows"], key=lambda r: r.get("row", 0))
        return [r.get("image") or r.get("path") or r.get("file") or "" for r in rows]
    for key in ("images", "paths", "files"):
        if isinstance(mapping.get(key), list):
            return [it if isinstance(it, str) else (it.get("path") or it.get("image") or it.get("file") or "")
                    for it in mapping[key]]
    raise ValueError("mapping.json has no rows/images/paths/files list")


class FlatIndex:
    """An in-memory, exact cosine index over page vectors with provenance."""

    def __init__(self, vectors: np.ndarray, filenames: list[str], provenance: dict[str, dict]):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vecs = (vectors / norms).astype(np.float32)  # unit vectors -> dot = cosine
        self.filenames = filenames
        # Join to provenance by filename STEM so a binarized .png matches a harvested
        # .jpg. Fragment id comes from provenance, else the "{id}__NN" filename prefix.
        self.stems = [Path(f).stem for f in filenames]
        self.provenance = {Path(k).stem: v for k, v in provenance.items()}
        # per-column crops are named F-xxxx__NN_cM; page stem is the part before "_c".
        self._page_stem = [re.split(r"_c\d+$", s)[0] for s in self.stems]
        self.fragments = [self.provenance.get(ps, {}).get("fragment_id") or s.split("__")[0]
                          for s, ps in zip(self.stems, self._page_stem)]
        self._object_cache: tuple[list[str], np.ndarray] | None = None

    # ---- construction ----
    @classmethod
    def load(cls, npy_path: str | Path, provenance_csv: str | Path) -> "FlatIndex":
        npy_path = Path(npy_path)
        vectors = np.load(npy_path)
        mapping = json.loads(npy_path.with_suffix(npy_path.suffix + ".mapping.json").read_text()
                             if (npy_path.parent / (npy_path.name + ".mapping.json")).exists()
                             else (npy_path.with_name(npy_path.stem + ".mapping.json")).read_text())
        filenames = [Path(p).name for p in _mapping_paths(mapping)]
        prov: dict[str, dict] = {}
        with Path(provenance_csv).open() as fh:
            for row in csv.DictReader(fh):
                prov[row["filename"]] = row
        return cls(vectors, filenames, prov)

    def __len__(self) -> int:
        return len(self.filenames)

    # ---- search ----
    def search(self, query_idx: int, k: int = 5) -> list[Hit]:
        sims = self.vecs @ self.vecs[query_idx]
        order = np.argsort(-sims)
        hits: list[Hit] = []
        for j in order:
            if j == query_idx:
                continue
            f = self.filenames[j]
            ov = (self.provenance.get(self._page_stem[j], {}).get("overview_url")
                  or f"https://fragmentarium.ms/overview/{self.fragments[j]}")
            hits.append(Hit(len(hits) + 1, float(sims[j]), f, self.fragments[j], ov))
            if len(hits) >= k:
                break
        return hits

    def index_of(self, filename: str) -> int:
        return self.filenames.index(filename)

    # ---- object-level (one vector per manuscript) ----
    def object_vectors(self) -> tuple[list[str], np.ndarray]:
        """One unit vector per object: the renormalized mean of its page unit-vectors.

        A manuscript (``fragment_id``) is the real unit of retrieval — a scholar asks
        "which *other books* are in this hand", not "which other leaf". Averaging the
        pages of an object cancels page-specific noise (recto/verso, damage,
        illumination, layout) and keeps the shared-hand signal, so object queries are
        both cleaner (higher-margin) and de-duplicated (one hit per manuscript instead
        of its several leaves crowding the top). We ignore, for now, that a single
        document may contain more than one hand.

        Returns ``(object_ids, vecs)`` with ``object_ids`` sorted and ``vecs`` an
        ``(n_objects, dim)`` array of unit vectors aligned to it. Cached.
        """
        if self._object_cache is None:
            ids = sorted(set(self.fragments))
            frags = np.asarray(self.fragments)
            mat = np.zeros((len(ids), self.vecs.shape[1]), dtype=np.float32)
            for i, oid in enumerate(ids):
                mat[i] = self.vecs[frags == oid].mean(axis=0)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._object_cache = (ids, (mat / norms).astype(np.float32))
        return self._object_cache

    def _resolve_object(self, query: str) -> str:
        """Accept a fragment id (``F-eadz``) or any of its page filenames."""
        ids, _ = self.object_vectors()
        if query in ids:
            return query
        if query in self.filenames:
            return self.fragments[self.filenames.index(query)]
        stem = re.split(r"_c\d+$", Path(query).stem)[0].split("__")[0]
        if stem in ids:
            return stem
        raise ValueError(f"no object matches query {query!r}")

    def search_object(self, query: str, k: int = 5) -> list[ObjectHit]:
        """Return the k nearest *objects* to the query object (mean-pooled → cosine)."""
        ids, ovecs = self.object_vectors()
        qid = self._resolve_object(query)
        qi = ids.index(qid)
        sims = ovecs @ ovecs[qi]
        order = np.argsort(-sims)
        frags = np.asarray(self.fragments)
        hits: list[ObjectHit] = []
        for j in order:
            if j == qi:
                continue
            oid = ids[j]
            members = np.nonzero(frags == oid)[0]
            ov = (self.provenance.get(self._page_stem[members[0]], {}).get("overview_url")
                  or f"https://fragmentarium.ms/overview/{oid}")
            hits.append(ObjectHit(len(hits) + 1, float(sims[j]), oid, len(members), ov))
            if len(hits) >= k:
                break
        return hits

    # ---- label-free evaluation ----
    def same_fragment_eval(self) -> dict:
        """Top-1 and mAP where 'relevant' = a different page of the same fragment.

        Queries whose fragment has only one harvested page are skipped (no positive).
        """
        frags = np.array(self.fragments)
        sims_all = self.vecs @ self.vecs.T
        np.fill_diagonal(sims_all, -np.inf)
        top1_hits = 0
        aps: list[float] = []
        n_q = 0
        for i in range(len(self)):
            rel = (frags == frags[i])
            rel[i] = False
            n_rel = int(rel.sum())
            if n_rel == 0:
                continue
            n_q += 1
            order = np.argsort(-sims_all[i])
            ranked_rel = rel[order]
            if ranked_rel[0]:
                top1_hits += 1
            # average precision
            hit, precisions = 0, []
            for rank, is_rel in enumerate(ranked_rel, 1):
                if is_rel:
                    hit += 1
                    precisions.append(hit / rank)
            aps.append(sum(precisions) / n_rel)
        return {
            "pages": len(self),
            "fragments": int(len(set(self.fragments))),
            "queries_with_positive": n_q,
            "top1": top1_hits / n_q if n_q else 0.0,
            "mAP": float(np.mean(aps)) if aps else 0.0,
        }

    def object_split_eval(self) -> dict:
        """Label-free *object-level* smoke test: does averaging preserve object identity?

        Split each multi-page object's pages into two halves, mean-pool each half into a
        unit vector, and check whether the two halves of one object land nearest each
        other among all half-vectors. High Top-1 means the pooled representation is a
        stable per-object signal rather than an artefact of any single leaf. This is the
        object-level analogue of :meth:`same_fragment_eval`; objects with <2 pages are
        skipped, and with few pages per object it is a weak (but honest) proxy — the
        true target, same-*hand* retrieval across objects, needs hand labels we do not
        have here.
        """
        frags = np.asarray(self.fragments)
        ids = [o for o in sorted(set(self.fragments)) if int((frags == o).sum()) >= 2]
        halves: list[np.ndarray] = []
        owner: list[str] = []
        for oid in ids:
            members = np.nonzero(frags == oid)[0]
            cut = len(members) // 2
            for part in (members[:cut], members[cut:]):
                h = self.vecs[part].mean(axis=0)
                nrm = float(np.linalg.norm(h)) or 1.0
                halves.append(h / nrm)
                owner.append(oid)
        if not halves:
            return {"objects_with_2plus_pages": 0, "halves": 0, "top1": 0.0}
        H = np.asarray(halves, dtype=np.float32)
        owners = np.asarray(owner)
        sims = H @ H.T
        np.fill_diagonal(sims, -np.inf)
        nn = np.argmax(sims, axis=1)
        return {
            "objects_with_2plus_pages": len(ids),
            "halves": len(halves),
            "top1": float((owners[nn] == owners).mean()),
        }
