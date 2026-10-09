"""CAD stage: part model, defects, rendering, dataset generation, measurement."""
import csv

import numpy as np
import pytest

from qi.cad.measure import measure_bore
from qi.cad.part import VISUAL_DEFECTS, ImpellerSpec, build_impeller, make_part, on_vane, tessellate
from qi.cad.render import render_views


def test_good_part_is_valid_solid():
    part = build_impeller(ImpellerSpec())
    assert part.val().isValid()
    assert 40_000 < part.val().Volume() < 80_000      # mm^3, sanity range for this design


@pytest.mark.parametrize("defect", VISUAL_DEFECTS)
def test_each_defect_changes_the_geometry(defect):
    spec = ImpellerSpec()
    ok_vol = build_impeller(spec).val().Volume()
    part, info = make_part(spec, defect, np.random.default_rng(3))
    assert part.val().isValid()
    assert info["defect"] == defect
    vol = part.val().Volume()
    if defect == "flash":
        assert vol > ok_vol          # flash ADDS material
    else:
        assert vol < ok_vol          # holes, cracks, chips REMOVE material


def test_bore_oversize_records_true_size():
    _, info = make_part(ImpellerSpec(), "bore_oversize", np.random.default_rng(0))
    assert info["bore_d_actual"] > ImpellerSpec().bore_d + 1


def test_vane_detection():
    spec = ImpellerSpec()
    assert not on_vane(spec, 0.0, 0.0, margin=0)       # centre is the bore, not a vane


def test_render_produces_realistic_image():
    rng = np.random.default_rng(0)
    spec = ImpellerSpec()
    v, f = tessellate(build_impeller(spec))
    imgs = render_views(v, f, spec.outer_r, 2, rng)
    assert len(imgs) == 2
    for im in imgs:
        assert im.shape == (300, 300) and im.dtype == np.uint8
        assert im.std() > 10                             # not a blank image
    assert np.abs(imgs[0].astype(int) - imgs[1].astype(int)).mean() > 3   # views differ


def test_generated_dataset_layout(tiny_dataset):
    train, _ = tiny_dataset
    ok = list((train / "ok_front").glob("*.jpeg"))
    bad = list((train / "def_front").glob("*.jpeg"))
    assert len(ok) == 12 and len(bad) == 12
    rows = list(csv.DictReader(open(train / "metadata.csv")))
    assert len(rows) == 24
    assert {r["defect_type"] for r in rows if r["label"] == "defective"} <= set(VISUAL_DEFECTS)


def test_bore_measurement_pass_and_fail():
    rng = np.random.default_rng(5)
    nominal = ImpellerSpec()
    results = {"ok": [], "oversize": []}
    for kind, defect in (("ok", None), ("oversize", "bore_oversize")):
        for _ in range(6):
            spec = nominal.jittered(rng)
            part, _ = make_part(spec, defect, rng)
            img = render_views(*tessellate(part), spec.outer_r, 1, rng)[0]
            results[kind].append(measure_bore(img, nominal)["status"])
    assert results["ok"].count("PASS") >= 4
    assert results["oversize"].count("FAIL") >= 5
