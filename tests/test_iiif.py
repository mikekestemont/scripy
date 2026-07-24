"""Offline tests for the Stage-0 IIIF client, against the real F-eadz fixture."""

from __future__ import annotations

import json
from pathlib import Path

from scripy.iiif import parse_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "F-eadz.manifest.json"


def load_fixture():
    return parse_manifest(json.loads(FIXTURE.read_text()), manifest_id="F-eadz")


def test_parses_v2_manifest():
    man = load_fixture()
    assert man.presentation_version == 2
    assert "MAG-P 64.19" in man.label
    assert len(man) == 2


def test_language_metadata_read():
    # F-eadz carries "Middle Dutch" in its metadata; the reader should surface it.
    man = load_fixture()
    assert man.language and "Dutch" in man.language


def test_canvas_image_service_and_request_url():
    man = load_fixture()
    cv0 = man.canvases[0]
    assert cv0.image_service == "https://fragmentarium.ms:443/loris/F-eadz/fol_1r.jp2"
    # IIIF Image API request URL is built from the service base.
    assert cv0.image_request("full") == (
        "https://fragmentarium.ms:443/loris/F-eadz/fol_1r.jp2/full/full/0/default.jpg"
    )
    assert cv0.image_request("1024,").endswith("/full/1024,/0/default.jpg")


def test_v3_manifest_normalises_to_same_shape():
    v3 = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": "https://example.org/m1",
        "type": "Manifest",
        "label": {"en": ["A v3 manuscript"]},
        "items": [
            {
                "id": "https://example.org/m1/canvas/1",
                "type": "Canvas",
                "label": {"none": ["f. 1r"]},
                "width": 2000,
                "height": 3000,
                "items": [
                    {
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "type": "Annotation",
                                "body": {
                                    "id": "https://example.org/iiif/img1/full/max/0/default.jpg",
                                    "type": "Image",
                                    "service": [
                                        {"id": "https://example.org/iiif/img1", "type": "ImageService3"}
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    man = parse_manifest(v3)
    assert man.presentation_version == 3
    assert man.label == "A v3 manuscript"
    assert len(man) == 1
    assert man.canvases[0].image_service == "https://example.org/iiif/img1"
    assert man.canvases[0].image_request("max").endswith("/full/max/0/default.jpg")
