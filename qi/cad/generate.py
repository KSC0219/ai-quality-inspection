"""Generate a labelled synthetic dataset from the CAD model.

Output layout matches the real Kaggle dataset, so every training script works
on either one:

    out_dir/
        ok_front/   cad_ok_00012_v2.jpeg ...
        def_front/  cad_crack_00031_v0.jpeg ...
        metadata.csv  (file, label, defect type, defect parameters, part dims)
        sample_part_ok.step / sample_part_<defect>.step  (open in any CAD tool)
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .part import VISUAL_DEFECTS, ImpellerSpec, export_step, make_part, tessellate
from .render import render_views

OK_DIR, DEF_DIR = "ok_front", "def_front"


def _make_one(args) -> list[dict]:
    part_id, defect, views, seed, out_dir, spec_dict = args
    rng = np.random.default_rng(seed)
    spec = ImpellerSpec(**spec_dict).jittered(rng)
    part, info = make_part(spec, defect, rng)
    verts, faces = tessellate(part)
    folder = Path(out_dir) / (OK_DIR if defect is None else DEF_DIR)
    rows = []
    for v, img in enumerate(render_views(verts, faces, spec.outer_r, views, rng)):
        name = f"cad_{defect or 'ok'}_{part_id:05d}_v{v}.jpeg"
        cv2.imwrite(str(folder / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"file": f"{folder.name}/{name}", "label": "ok" if defect is None else "defective",
                     "defect_type": info["defect"], "part_id": part_id, "view": v,
                     "defect_params": json.dumps({k: x for k, x in info.items() if k != "defect"}),
                     "outer_d": round(spec.outer_d, 2), "bore_d": round(spec.bore_d, 2)})
    return rows


def generate_dataset(out_dir: str | Path, n_ok_parts: int = 150, n_def_parts: int = 150,
                     views_per_part: int = 4, defect_types: list[str] | None = None,
                     seed: int = 42, workers: int | None = None, spec: ImpellerSpec | None = None,
                     export_samples: bool = True, progress: bool = True) -> Path:
    """Create n_ok_parts good + n_def_parts defective parts, each photographed
    views_per_part times. Defect types are spread evenly across defective parts."""
    out_dir = Path(out_dir)
    (out_dir / OK_DIR).mkdir(parents=True, exist_ok=True)
    (out_dir / DEF_DIR).mkdir(parents=True, exist_ok=True)
    defect_types = defect_types or VISUAL_DEFECTS
    spec = spec or ImpellerSpec()
    seeds = np.random.SeedSequence(seed).spawn(n_ok_parts + n_def_parts)
    jobs = [(i, None, views_per_part, s.generate_state(1)[0], str(out_dir), spec.to_dict())
            for i, s in enumerate(seeds[:n_ok_parts])]
    jobs += [(n_ok_parts + i, defect_types[i % len(defect_types)], views_per_part,
              s.generate_state(1)[0], str(out_dir), spec.to_dict())
             for i, s in enumerate(seeds[n_ok_parts:])]

    rows: list[dict] = []
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1:
        # "spawn" = fresh worker processes. Forking a process that already
        # loaded the CAD kernel (OpenCASCADE) can deadlock.
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for k, r in enumerate(ex.map(_make_one, jobs, chunksize=4)):
                rows += r
                if progress and (k + 1) % 25 == 0:
                    print(f"  {k + 1}/{len(jobs)} parts done")
    else:
        for k, job in enumerate(jobs):
            rows += _make_one(job)
            if progress and (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)} parts done")

    with open(out_dir / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if export_samples:  # real CAD files you can open and show in an interview
        rng = np.random.default_rng(seed)
        for d in [None] + list(defect_types):
            part, _ = make_part(spec, d, rng)
            export_step(part, str(out_dir / f"sample_part_{d or 'ok'}.step"))
    if progress:
        n_ok = sum(r["label"] == "ok" for r in rows)
        print(f"Synthetic dataset: {n_ok} OK + {len(rows) - n_ok} defective images in {out_dir}")
    return out_dir
