# scripy — Blueprint

*A streaming handwriting search index built over IIIF, powered by the `mole` encoder.*

---

## 0. What scripy is (and what it is not)

`scripy` turns the world's IIIF image collections into a **searchable index of
handwriting**: give it a query page (or a fragment ID) and it returns the pages,
across every harvested collection, written by a visually similar hand.

It does this **without ever building a permanent image archive**. Images are
streamed from IIIF servers, encoded into a fixed-dimensional vector, and deleted.
Only the vectors and their provenance are kept.

`scripy` deliberately owns **none** of the machine learning. The encoder, the
descriptor extraction, the VLAD codebook, and the retrieval metrics all come from
the sibling project [`mole`](../../mole). The division of labour is strict:

| Concern | Owner |
|---|---|
| Trained ViT encoder + checkpoints | `mole` |
| Patch-window foreground descriptors | `mole` (`mole.embed`) |
| Frozen universal VLAD codebook (K=100) | `mole` (`mole codebook`) |
| Retrieval metrics (mAP, macro-mAP, Top-k) | `mole` (`mole eval`) |
| **IIIF discovery + harvesting** | **scripy** |
| **Streaming / bounded-memory orchestration** | **scripy** |
| **State, checkpointing, restartability** | **scripy** |
| **Vector index (ANN) + search API/UI** | **scripy** |

Rule of thumb: if a line of code touches a neural network or a k-means centroid,
it belongs in `mole` and scripy imports it. scripy is the plumbing that lets that
model see the whole IIIF universe one page at a time.

---

## 1. Guiding principles

1. **Prototype first, grow in complexity.** Every stage in §3 is runnable and
   independently useful. Stage 0 is a walking skeleton over a *single* manifest;
   the streaming machinery, the scale-out crawl, and the continual-learning parts
   only appear once the simple thing works end to end. We never build the daemon
   before we can index one page.
2. **Reuse `mole`; never reinvent it.** No duplicated descriptor / codebook /
   metric code. scripy calls `mole` as a library (see §7).
3. **Freeze the codebook.** This is the single biggest simplification versus a
   naive design — see §8. A fixed embedding space means a newly harvested page
   gets a vector with *zero* global recomputation. It removes the shadow-centroid
   machinery, the promotion gating, and the replay buffer from the v1 hot path
   entirely.
4. **Stream, don't archive.** Bounded queues, fixed-size batches, images deleted
   immediately after a successful index write. Memory is bounded by queue depth,
   not corpus size.
5. **Everything is restartable from SQLite.** A crawl that dies at page 200,000
   resumes at 200,001. No stage recomputes completed work.
6. **CPU and GPU are never idle simultaneously.** Downloading and decoding (CPU /
   network) overlap with encoding (GPU) through the queue structure in §6.
7. **Politeness is a feature, not an afterthought.** IIIF servers are run by
   libraries, not hyperscalers. Rate-limit per host, honour `Retry-After`, set a
   descriptive `User-Agent`, cache `info.json`. A harvester that gets scripy
   blocked from Fragmentarium has failed regardless of its throughput.

---

## 2. The case study: Fragmentarium's Middle Dutch fragments

We start narrow, on purpose. The first corpus is the **Middle Dutch** material in
[Fragmentarium](https://fragmentarium.ms), the Fribourg digital laboratory for
medieval manuscript fragments. It is the ideal shakedown corpus:

- **Small and bounded.** Dozens of fragments, most only a leaf or two — the whole
  thing fits on a laptop and streams in minutes, so the prototype loop is fast.
- **In-domain.** Middle Dutch charters and literary fragments are the same
  language/region `mole` was trained on (Antwerp, Flanders, Utrecht, Leroy).
  Retrieval quality should be strong from day one, which keeps early debugging
  about *plumbing*, not *model failure*.
- **Genuinely IIIF-native.** Every fragment ships a clean IIIF manifest, so it
  exercises the real harvesting path rather than a bespoke downloader.
- **A real scholarly payoff.** Fragmentarium's raison d'être is *reassembling
  dispersed fragments of the same parent manuscript*. Same-hand retrieval is a
  direct tool for that — e.g. the F-eadz "Middle Dutch Alexander compilation" is
  attributed to the Ferguut scribe; scripy should surface other leaves in that
  hand across collections.

### Verified IIIF facts (as of harvest bootstrap)

| Thing | Value |
|---|---|
| Presentation API | **v2** (`@context: .../presentation/2/context.json`, `@type: sc:Manifest`) |
| Manifest URL | `https://fragmentarium.ms/metadata/iiif/{FRAGMENT_ID}/manifest.json` |
| Human overview page | `https://fragmentarium.ms/overview/{FRAGMENT_ID}` |
| Image API | **v2, level2**, Loris server |
| Image service base | `https://fragmentarium.ms:443/loris/{FRAGMENT_ID}/{IMAGE}.jp2` |
| Full-image request | `{service}/full/{size}/0/default.jpg` |
| Language signal | manifest `metadata[].label == "Title"` / a `"Language"` field carries `"Middle Dutch"` |

Reference fragment: **F-eadz** — *Antwerpen, Universiteitsbibliotheek Antwerpen
Bijzondere Collecties, MAG-P 64.19*, "Middle Dutch Alexander compilation",
2 canvases. Its manifest is checked in as a test fixture
(`tests/fixtures/F-eadz.manifest.json`).

### Discovery is the one thing Fragmentarium does *not* hand us

Fragmentarium exposes per-fragment manifests but (as of writing) no documented
public JSON API to *enumerate* the Middle Dutch subset. The staged answer:

- **Stage 0–1:** work from a **hand-curated seed list** of fragment IDs
  (`data/seeds/middle-dutch.txt`). Enough to build and validate the whole pipeline.
- **Stage 3:** automate discovery — scrape the `fragmentarium.ms/search` results
  for the Middle Dutch language facet, and/or consume a IIIF **Collection**
  manifest if one is exposed. Treat this as a pluggable `Discoverer` interface so
  each repository (Fragmentarium, e-codices, Digital Bodleian, …) supplies its own.

> **Open question for the Fragmentarium team:** is there a IIIF Collection endpoint
> or a queryable API for the language facet? If so, Stage-3 discovery collapses to
> a single manifest fetch. (Added to the running questions list.)

---

## 3. Staged roadmap (prototype → system)

Each stage ends with a concrete **Done when** gate. Do not start a stage before
the previous gate is green.

### Stage 0 — One manifest, end to end *(the walking skeleton)*
- **Goal:** `scripy` fetches the F-eadz manifest, lists its canvases, builds image
  request URLs, downloads one page to a temp file, and hands it to `mole embed` to
  get a single VLAD vector. Print the vector shape and exit.
- **New:** the IIIF manifest client (`scripy.iiif`), the mole bridge.
- **No:** database, queues, threads, deletion policy — all inlined/skipped.
- **Done when:** `scripy manifest F-eadz` prints 2 canvases with correct image
  URLs (offline, from the fixture), and the online path yields one `(1, 38400)`
  VLAD vector. *This is the only stage that must exist before anything else.*

### Stage 1 — The Middle Dutch mini-corpus + a searchable index
- **Goal:** harvest every fragment in the seed list, encode each page, build an
  in-memory / flat vector index, and answer "given canvas X, return the k nearest
  hands." Wire `mole eval` in as the quality gate on the labelled subset.
- **New:** batch harvest over a seed list; a flat (numpy / exact) index; a
  `scripy search` command; provenance rows (fragment ID, canvas ID, IIIF URL).
- **Done when:** a leave-one-out retrieval run over the Middle Dutch seed corpus
  reports a sane macro-mAP via `mole eval`, and `scripy search <canvas>` returns
  plausible same-hand neighbours with clickable Fragmentarium links.

### Stage 2 — Streaming, bounded, restartable
- **Goal:** turn the batch script into the real streaming pipeline. Introduce
  SQLite state, bounded queues, the temp SSD cache with eviction, and
  **delete-after-successful-index**. Kill it mid-run; restart; it resumes.
- **New:** the architecture of §4–§6 (scheduler, download pool, preprocess pool,
  GPU encoder, writer), SQLite schema (§5), checkpointing after every batch.
- **Done when:** re-running after a `kill -9` re-indexes zero already-done canvases,
  peak RSS is flat regardless of corpus size, and the temp cache never exceeds its
  configured cap.

### Stage 3 — Scale-out crawl
- **Goal:** widen from the seed list to **all of Fragmentarium**, then to a second
  repository, via the pluggable `Discoverer`. FAISS replaces the flat index.
- **New:** automated discovery (search-facet / IIIF Collection), per-host rate
  limiting & politeness, memory-mapped FAISS (half precision), a manifest DB that
  spans repositories.
- **Done when:** a multi-repository crawl runs unattended for hours within a fixed
  memory budget, and search returns cross-collection same-hand hits.

### Stage 4 — Serving
- **Goal:** a thin query service. Upload/point-to a page → k nearest hands with
  thumbnails (fetched on demand via the IIIF Image API — still no stored images)
  and deep links back to each source viewer.
- **New:** a small HTTP API + minimal UI; on-demand IIIF thumbnail proxying.
- **Done when:** a non-technical scholar can paste a Fragmentarium URL and get a
  ranked, linked result page.

### Stage 5 — Continual maintenance *(optional, deferred)*
- **Goal:** keep the index fresh as new collections arrive **without** letting the
  embedding space drift out from under the stored vectors.
- **New:** codebook drift monitoring; **scheduled offline re-fit + re-embed**,
  promoted only if a held-out `mole eval` benchmark does not regress (§8). Truly
  online / incremental VLAD and a replay buffer live here and *only* here.
- **Done when:** a codebook refresh can be promoted or rolled back on evidence,
  with the old index kept until the new one is proven no worse.

---

## 4. Architecture

```
                 ┌─────────────┐
   Discoverer ──►│ Manifest DB │  (SQLite: repositories, manifests, canvases, jobs)
   (per repo)    └──────┬──────┘
                        │ pending canvases
                        ▼
                 Download queue ──► Download pool (async HTTP, IIIF Image API)
                        │                 │  temp SSD cache (JPEG), auto-evicted
                        ▼                 ▼
                 Preprocess queue ──► Preprocess pool  ── mole: decode → (binarize/
                        │                                  zones/foreground) → patch-
                        │                                  window descriptors
                        ▼
                 Descriptor queue ──► GPU encoder ── mole: frozen-codebook VLAD
                        │                             → one page vector
                        ▼
                    Writer ──►  Vector index (FAISS, mmap, fp16)
                        └──────►  SQLite: canvas → vector-id, provenance, checkpoint
                        └──────►  delete temp image (only after a successful write)
```

Differences from the original blueprint, and why:

- **"Line segmentation" → mole's actual page pipeline.** `mole` retrieves at the
  **page/document** level using patch-window foreground-token descriptors (with an
  optional binarize/zone step), not word/line spotting. The retrieval unit is a
  page vector, which is exactly what `mole eval` scores. Line-level granularity is
  a possible *future* index, not a v1 requirement.
- **"VLAD Encoder (GPU) + Incremental VLAD" → frozen codebook.** See §8. The GPU
  stage applies a *fixed* codebook; it does not learn one online in v1.
- **Explicit temp cache + deletion edge.** Deletion is an event on the writer, not
  a background sweep, so "indexed" and "deleted" cannot disagree after a crash.

---

## 5. Storage & state

**SQLite** (the source of truth for *what has been done*):

- `repository(id, name, base_url, discoverer, politeness_json)`
- `manifest(id, repository_id, iiif_url, label, language, discovered_at, status)`
- `canvas(id, manifest_id, iiif_canvas_id, image_service_url, width, height, status)`
  — `status ∈ {pending, downloaded, encoded, indexed, failed}`
- `job(id, canvas_id, stage, attempts, last_error, updated_at)` — restart bookkeeping
- `vector(canvas_id, faiss_id, codebook_version, model_id, dim)` — provenance link
- `checkpoint(key, value)` — crawl cursor, codebook version, counters

**Vector index:** flat numpy at Stage 1 → **FAISS** at Stage 3 (memory-mapped,
`fp16` storage, cosine / inner-product on L2-normalised vectors). The stored
`model_id` and `codebook_version` are non-negotiable provenance: a vector is only
comparable to others sharing both.

**Temp cache:** a size-capped SSD directory for in-flight JPEGs, evicted LRU and
truncated hard at a configured ceiling. Images here are the *only* pixel data
scripy ever persists, and only until their page is indexed.

---

## 6. Concurrency model

1. **Scheduler (1 thread):** pulls `pending` canvases from SQLite, fills the
   download queue, respects per-host rate limits.
2. **Download pool (2–4 threads):** async HTTP only (keep-alive, `Retry-After`).
   One IIIF Image API request per canvas at a target size.
3. **Preprocess pool (N_cpu − 2):** JPEG decode once, optional binarize/zone/
   foreground, patch-window descriptor extraction via `mole`.
4. **GPU encoder (1 worker):** accumulates descriptor batches, applies the frozen
   VLAD codebook, emits page vectors. Adaptive batch size by GPU memory.
5. **Writer (1 thread):** FAISS + SQLite writes, then deletes the temp image.

All queues are **bounded**; a full queue back-pressures the stage upstream, which
is what keeps memory flat. Note from `mole` experience: the *training* aug pipeline
is CPU-bound, but **inference** here is far lighter (no multi-crop augmentation) —
still, size the preprocess pool to keep the GPU encoder fed, and profile before
adding GPUs (on a CPU-starved box a second GPU can go *backwards*).

---

## 7. The mole bridge

scripy imports `mole` rather than shelling out, to keep the streaming path in one
process and one GPU context. The pieces it reuses (already built and validated in
`mole`):

- `load_backbone(checkpoint)` — load teacher weights into the canonical ViT.
- the embed path's `_build_transform` / `_page_tokens` / `patch_descriptors` /
  foreground mask — per-window foreground-token descriptors.
- `vlad_encode(..., codebook)` with **`--codebook-from`** semantics — apply a
  frozen codebook saved as `<name>.codebook.npy`.
- `mole eval` — leave-one-out mAP / macro-mAP / Top-k as the retrieval-quality gate.

**Pinned artifacts scripy depends on:** a single checkpoint (the pooled
multi-archive finetune) and one `universal.codebook.npy` (K=100), both produced by
`mole` and referenced by `model_id` + `codebook_version`. scripy treats these as
immutable inputs; regenerating them is a `mole` task, not a scripy task.

Install: `pip install -e ../mole` alongside scripy (see README). If the full mole
training stack is undesirable in a deployment, the fallback is the
"call `mole embed` as a CLI" mode — looser, slower, but zero shared process state.

---

## 8. Codebook policy (the reframed "incremental VLAD")

The original blueprint made an **online, self-updating codebook** — shadow
centroids, a promotion gate, a 1–5M-descriptor replay buffer — a *core v1 feature*.
For this system that is the wrong default, and `mole`'s own experiments say so:

- The **frozen universal VLAD codebook (K=100)** is already the settled index
  descriptor in `mole`. It is robust: it survived *unseen* writers on Historical-WI
  and degrades only gracefully across domains. Fitting on all pooled charters vs a
  25% sample changed retrieval by ~0.001 — the codebook is a **vocabulary of local
  stroke shapes**, not a set of per-writer slots, so it saturates quickly and does
  not need every new hand to re-fit.
- The measured cost of freezing (vs a per-collection transductive codebook) is
  about **−0.03 macro-mAP on average**, and Top-1 is essentially untouched. That
  buys three things a moving codebook cannot: (a) one comparable vector space
  across all collections, (b) a new page indexed with **zero global recompute**,
  and (c) no per-archive re-fit. For a search index, that trade is decisively worth
  it. Raising K to 256 was measured **worse**, so K=100 stays.

**Therefore, v1 uses a frozen codebook and no online codebook learning at all.**
This deletes an enormous amount of the original machinery from the hot path.

The one true hazard of a fixed space is **corpus drift**: a genuinely
out-of-distribution new collection (a different script, a different medium) whose
stroke vocabulary the K=100 codebook underserves. The response is deferred to
Stage 5 and kept **offline**:

1. Monitor drift (mean assignment distance to nearest centroid per new batch).
2. When drift crosses a budget, schedule an **offline re-fit** of the codebook over
   the enlarged corpus and a **full re-embed**.
3. **Promote only on evidence:** keep the old index; build the new one; promote
   iff a held-out `mole eval` benchmark does not regress. Roll back otherwise.

This preserves the original's best idea (never promote a codebook that regresses
retrieval) while moving it off the per-page path, where it does not belong. A truly
online codebook + replay buffer remains possible, but it is a *last* milestone, not
a *first* one — and it always carries the re-embed cost, because a moving codebook
means a moving embedding space and therefore stale stored vectors.

---

## 9. Benchmarks

Two families, tracked continuously:

**Retrieval quality** (via `mole eval`, on the labelled subset — the gate that
decides whether a change ships): mAP, **macro-mAP** (the honest number under
class imbalance), Top-1 / Top-5 / Top-10, and `--cross-doc-only` to strip the
same-document scan shortcut.

**System throughput** (the gate that decides whether the crawl is healthy):
pages/hour, descriptors/second, GPU utilisation, peak RSS, temp-cache occupancy,
download error/retry rate, and codebook drift.

A change to the harvesting/index layer must not move retrieval quality (same
`model_id` + `codebook_version` ⇒ identical vectors). A change to the model/codebook
is a `mole` release with a new version stamp and a full re-embed.

---

## 10. Risks & open questions

- **Fragmentarium discovery API.** No documented public enumeration endpoint yet
  (§2). Mitigation: seed list now, pluggable `Discoverer` later. *Ask the team.*
- **IIIF version spread.** Fragmentarium is Presentation **v2**; the wider crawl
  will hit **v3** (`items`/`AnnotationPage`) and Image API v3. The manifest client
  must normalise both from Stage 0 so v3 is not a rewrite later.
- **Image-server politeness.** Loris/other servers can be slow or rate-limited.
  Per-host limits, `Retry-After`, cached `info.json`, descriptive `User-Agent`.
- **Retrieval granularity.** Page-level vectors match `mole` today. Fragment
  reassembly might eventually want line- or column-level vectors; keep the canvas →
  vector mapping one-to-many-capable in the schema so that is additive.
- **Provenance integrity.** Every vector must carry `model_id` + `codebook_version`;
  cross-version comparison is a bug. Enforced in the `vector` table.
- **Licensing / terms.** Harvesting metadata + streaming images for indexing is not
  redistribution, and no images are stored — but record each repository's terms per
  `repository` row and honour opt-outs.

---

*This document is the living design. Stage gates in §3 are the definition of done;
`mole` is the model of record; the codebook is frozen until §8 says otherwise.*
