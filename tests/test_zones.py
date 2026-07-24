"""Tests for IIIF pct: region-URL construction from YOLO bboxes."""

from __future__ import annotations

from scripy.zones import _column_boxes, _pct_region_url, _service_base


def test_column_boxes_merges_overlapping_text_and_textmain():
    # two real columns, each detected as overlapping Text + Text_Main; a stamp is ignored
    dets = [
        ["Text", 0.94, 800, 30, 1330, 450],
        ["Text_Main", 0.94, 810, 40, 1327, 451],   # ~duplicate of the above column
        ["Text_Main", 0.93, 210, 10, 730, 400],
        ["Text", 0.92, 211, 11, 728, 401],          # ~duplicate of the left column
        ["Marks_Stamp", 0.89, 686, 182, 823, 317],  # not text -> dropped
        ["Text", 0.30, 0, 0, 50, 50],               # below min_conf -> dropped
    ]
    cols = _column_boxes(dets, min_conf=0.5)
    assert len(cols) == 2                    # two columns, duplicates merged
    assert cols[0][0] < cols[1][0]           # sorted left-to-right by x0
    assert (210, 10, 730, 400) in cols       # left column kept (highest conf of its pair)


def test_service_base_strips_image_request():
    url = "https://frag.ms:443/loris/F-eadz/fol_1r.jp2/full/1400,/0/default.jpg"
    assert _service_base(url) == "https://frag.ms:443/loris/F-eadz/fol_1r.jp2"


def test_pct_region_no_padding():
    # bbox covers x 350..700 of 1400 wide, y 200..1000 of 2000 tall
    url = _pct_region_url("SVC", [350, 200, 700, 1000], [1400, 2000], size="full", pad_pct=0.0)
    assert url == "SVC/pct:25.000,10.000,25.000,40.000/full/0/default.jpg"


def test_pct_region_padding_clamps_to_bounds():
    # a bbox flush against the top-left with padding must not go negative
    url = _pct_region_url("SVC", [0, 0, 700, 1000], [1400, 2000], size="full", pad_pct=2.0)
    # X,Y clamp to 0; W/H grow by pad but stay within 100
    assert "/pct:0.000,0.000," in url
    body = url.split("pct:")[1].split("/")[0]
    x, y, w, h = (float(v) for v in body.split(","))
    assert x == 0.0 and y == 0.0
    assert x + w <= 100.0 and y + h <= 100.0
