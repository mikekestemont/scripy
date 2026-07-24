"""Flat vector index + search (BLUEPRINT §3 Stage 1; FAISS replaces this at Stage 3).

Loads the page vectors mole wrote (``.npy`` + ``.mapping.json``), joins them to the
harvest ``provenance.csv``, and answers cosine nearest-neighbour queries. Also
provides a label-free sanity metric: *same-fragment retrieval* — can a page find the
other leaves of its own fragment? With Fragmentarium's dispersed fragments this is a
lower bound on same-hand retrieval (leaves of one manuscript are often catalogued as
separate fragments), but it needs no hand labels and exercises the whole pipeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FlatIndex", "Hit"]


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    filename: str
    fragment_id: str
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
        self.fragments = [self.provenance.get(s, {}).get("fragment_id") or s.split("__")[0]
                          for s in self.stems]

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
            hits.append(Hit(len(hits) + 1, float(sims[j]), f,
                            self.fragments[j], self.provenance.get(self.stems[j], {}).get("overview_url", "")))
            if len(hits) >= k:
                break
        return hits

    def index_of(self, filename: str) -> int:
        return self.filenames.index(filename)

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
