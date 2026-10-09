"""Validate the CAD dimension check on parts with KNOWN bore sizes.

Generates good parts (bore 12 mm) and parts whose bore was machined too large
(13.2-15 mm), measures every photo and reports accuracy.

    python scripts/measure_check.py --parts 50
Writes results/measurement_check.json and results/measurement_examples.png
"""
import argparse
import json

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from qi.cad.measure import draw_measurement, measure_bore
from qi.cad.part import ImpellerSpec, make_part, tessellate
from qi.cad.render import render_views
from qi.config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=50, help="per class (good / oversize)")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    nominal = ImpellerSpec()
    rows, examples = [], []
    for defect in [None] * a.parts + ["bore_oversize"] * a.parts:
        spec = nominal.jittered(rng)
        part, info = make_part(spec, defect, rng)
        img = render_views(*tessellate(part), spec.outer_r, 1, rng)[0]
        r = measure_bore(img, nominal)
        # true bore expressed in the measurement's scale (hub = 30 mm ruler)
        true = info.get("bore_d_actual", spec.bore_d) * nominal.hub_d / spec.hub_d
        rows.append({"truth": "oversize" if defect else "good", "true_bore_mm": round(true, 2),
                     "status": r["status"], "measured_mm": r.get("bore_mm")})
        if len(examples) < 4 and (defect is None) == (len(examples) < 2):
            examples.append(draw_measurement(img, r))

    measured = [x for x in rows if x["status"] in ("PASS", "FAIL")]
    err = np.array([abs(x["measured_mm"] - x["true_bore_mm"]) for x in measured])
    good = [x for x in rows if x["truth"] == "good"]
    over = [x for x in rows if x["truth"] == "oversize"]
    report = {
        "parts": len(rows),
        "median_abs_error_mm": round(float(np.median(err)), 3),
        "p90_abs_error_mm": round(float(np.percentile(err, 90)), 3),
        "within_0.5mm": round(float(np.mean(err <= 0.5)), 3),
        "good_parts_passed": f"{sum(x['status'] == 'PASS' for x in good)}/{len(good)}",
        "oversize_parts_failed": f"{sum(x['status'] == 'FAIL' for x in over)}/{len(over)}",
        "uncertain": sum(x["status"] == "UNCERTAIN" for x in rows),
        "not_found": sum(x["status"] == "NOT_FOUND" for x in rows),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "measurement_check.json").write_text(json.dumps({"summary": report, "rows": rows}, indent=2))
    cv2.imwrite(str(RESULTS_DIR / "measurement_examples.png"),
                cv2.cvtColor(np.hstack(examples), cv2.COLOR_RGB2BGR))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
