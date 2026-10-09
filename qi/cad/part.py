"""Parametric CAD model of a cast pump impeller, plus casting-defect injection.

The real dataset (Kaggle "casting product image data for quality inspection")
shows top views of a submersible-pump impeller. This module builds a similar
part as a solid CAD model with CadQuery (an open-source, Python-scripted CAD
kernel built on OpenCASCADE, the same family of B-rep kernel used by commercial
CAD tools), then creates defective versions by boolean operations:

    blowhole      sphere(s) cut into the surface      (gas porosity)
    pinholes      cluster of tiny spheres             (fine porosity)
    crack         thin zig-zag slot cut into the top  (hot tear / crack)
    edge_chip     cylinder cut out of the rim         (broken edge)
    flash         thin plate added at the rim         (flash / burr)
    bore_oversize bore machined too large             (dimensional defect,
                                                       caught by measure.py)

Every part gets small random variation (+/-0.3 mm) like real castings.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

try:
    import cadquery as cq
except ImportError as e:  # pragma: no cover
    raise ImportError("CadQuery is needed for the CAD stage: pip install cadquery") from e

VISUAL_DEFECTS = ["blowhole", "pinholes", "crack", "edge_chip", "flash"]
ALL_DEFECTS = VISUAL_DEFECTS + ["bore_oversize"]


@dataclass(frozen=True)
class ImpellerSpec:
    """Nominal design dimensions in millimetres (the 'CAD drawing' values)."""
    outer_d: float = 100.0      # disc diameter
    disc_t: float = 6.0         # disc thickness
    hub_d: float = 30.0         # hub diameter
    hub_h: float = 10.0         # hub height above the disc
    bore_d: float = 12.0        # shaft bore diameter (toleranced feature)
    bore_tol: float = 0.5       # +/- tolerance on the bore
    vane_count: int = 7
    vane_h: float = 5.0         # vane height above the disc
    vane_t: float = 3.0         # vane thickness
    vane_sweep_deg: float = 55  # how much each vane curves from hub to rim

    @property
    def outer_r(self) -> float:
        return self.outer_d / 2

    def jittered(self, rng: np.random.Generator, mm: float = 0.3) -> "ImpellerSpec":
        """Same design with manufacturing variation, like any real casting."""
        j = lambda v: float(v + rng.uniform(-mm, mm))  # noqa: E731
        return replace(self, outer_d=j(self.outer_d), disc_t=j(self.disc_t), hub_d=j(self.hub_d),
                       vane_t=j(self.vane_t), vane_h=j(self.vane_h),
                       bore_d=float(self.bore_d + rng.uniform(-0.1, 0.1)),
                       vane_sweep_deg=float(self.vane_sweep_deg + rng.uniform(-4, 4)))

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- geometry
def _vane_centerline(spec: ImpellerSpec, i: int, n_pts: int = 14) -> np.ndarray:
    r0, r1 = spec.hub_d / 2 - 1.0, spec.outer_r - 3.0
    base = 2 * math.pi * i / spec.vane_count
    rs = np.linspace(r0, r1, n_pts)
    th = base + math.radians(spec.vane_sweep_deg) * (rs - r0) / (r1 - r0)
    return np.stack([rs * np.cos(th), rs * np.sin(th)], axis=1)


def _vane_polygon(spec: ImpellerSpec, i: int) -> list[tuple[float, float]]:
    c = _vane_centerline(spec, i)
    d = np.gradient(c, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    half = spec.vane_t / 2
    left, right = c + n * half, c - n * half
    return [tuple(p) for p in np.vstack([left, right[::-1]])]


def on_vane(spec: ImpellerSpec, x: float, y: float, margin: float = 1.0) -> bool:
    """True if point (x, y) on the disc lies under/next to a vane."""
    p = np.array([x, y])
    for i in range(spec.vane_count):
        if np.min(np.linalg.norm(_vane_centerline(spec, i, 40) - p, axis=1)) < spec.vane_t / 2 + margin:
            return True
    return False


def build_impeller(spec: ImpellerSpec, bore_d: float | None = None) -> cq.Workplane:
    """Build the defect-free solid: disc + hub + curved vanes + shaft bore."""
    part = cq.Workplane("XY").circle(spec.outer_r).extrude(spec.disc_t)
    part = part.edges(">Z").chamfer(0.6)  # cast parts have broken edges
    hub = cq.Workplane("XY").workplane(offset=spec.disc_t - 0.1).circle(spec.hub_d / 2).extrude(spec.hub_h + 0.1)
    part = part.union(hub)
    for i in range(spec.vane_count):
        vane = (cq.Workplane("XY").workplane(offset=spec.disc_t - 0.1)
                .polyline(_vane_polygon(spec, i)).close().extrude(spec.vane_h + 0.1))
        part = part.union(vane)
    bore = cq.Workplane("XY").workplane(offset=-1).circle((bore_d or spec.bore_d) / 2).extrude(
        spec.disc_t + spec.hub_h + 2)
    return part.cut(bore)


# ---------------------------------------------------------------- defects
def _random_disc_point(spec: ImpellerSpec, rng: np.random.Generator, r_min: float, r_max: float,
                       avoid_vanes: bool = True) -> tuple[float, float]:
    for _ in range(200):
        r, t = rng.uniform(r_min, r_max), rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(t), r * math.sin(t)
        if not avoid_vanes or not on_vane(spec, x, y):
            return x, y
    return x, y


def _sphere(x, y, z, r):
    return cq.Workplane("XY").sphere(r).translate((x, y, z))


def _blowhole(part, spec, rng):
    n = int(rng.integers(1, 4))
    holes = []
    for _ in range(n):
        x, y = _random_disc_point(spec, rng, spec.hub_d / 2 + 4, spec.outer_r - 4)
        r = float(rng.uniform(1.2, 3.0))
        part = part.cut(_sphere(x, y, spec.disc_t + 0.35 * r, r))
        holes.append({"x": round(x, 2), "y": round(y, 2), "r": round(r, 2)})
    return part, {"holes": holes}


def _pinholes(part, spec, rng):
    cx, cy = _random_disc_point(spec, rng, spec.hub_d / 2 + 6, spec.outer_r - 6)
    n = int(rng.integers(6, 14))
    for _ in range(n):
        x, y = cx + rng.normal(0, 2.5), cy + rng.normal(0, 2.5)
        r = float(rng.uniform(0.6, 1.1))
        part = part.cut(_sphere(x, y, spec.disc_t + 0.1 * r, r))
    return part, {"center": [round(cx, 2), round(cy, 2)], "count": n}


def _crack(part, spec, rng):
    x, y = _random_disc_point(spec, rng, spec.hub_d / 2 + 5, spec.outer_r - 3, avoid_vanes=False)
    ang = rng.uniform(0, 2 * math.pi)
    segs = int(rng.integers(2, 5))
    width = float(rng.uniform(0.8, 1.4))
    depth = float(rng.uniform(2.2, 3.5))
    total = 0.0
    for _ in range(segs):
        length = float(rng.uniform(4, 9))
        ang += rng.uniform(-0.6, 0.6)
        mx, my = x + math.cos(ang) * length / 2, y + math.sin(ang) * length / 2
        slot = (cq.Workplane("XY").box(length + width, width, depth + 2)
                .rotate((0, 0, 0), (0, 0, 1), math.degrees(ang))
                .translate((mx, my, spec.disc_t - depth / 2 + 1)))
        part = part.cut(slot)
        x, y = x + math.cos(ang) * length, y + math.sin(ang) * length
        total += length
    return part, {"length_mm": round(total, 1), "width_mm": round(width, 2)}


def _edge_chip(part, spec, rng):
    t = rng.uniform(0, 2 * math.pi)
    rc = float(rng.uniform(3.0, 7.0))
    d = spec.outer_r + rc * rng.uniform(0.2, 0.55)
    cut = (cq.Workplane("XY").workplane(offset=-1).circle(rc).extrude(spec.disc_t + spec.vane_h + 2)
           .translate((d * math.cos(t), d * math.sin(t), 0)))
    return part.cut(cut), {"angle_deg": round(math.degrees(t), 1), "chip_r": round(rc, 2)}


def _flash(part, spec, rng):
    t0 = rng.uniform(0, 2 * math.pi)
    span = rng.uniform(0.25, 0.6)                    # radians of rim covered
    ts = np.linspace(t0, t0 + span, 9)
    out = spec.outer_r + rng.uniform(1.0, 3.0, size=ts.size)
    inner = [(spec.outer_r - 1) * np.array([math.cos(t), math.sin(t)]) for t in ts]
    outer = [o * np.array([math.cos(t), math.sin(t)]) for o, t in zip(out, ts)]
    poly = [tuple(p) for p in inner] + [tuple(p) for p in outer[::-1]]
    th = float(rng.uniform(0.6, 1.4))
    plate = (cq.Workplane("XY").workplane(offset=spec.disc_t / 2 - th / 2)
             .polyline(poly).close().extrude(th))
    return part.union(plate), {"angle_deg": round(math.degrees(t0), 1), "span_deg": round(math.degrees(span), 1)}


_DEFECT_FUNCS = {"blowhole": _blowhole, "pinholes": _pinholes, "crack": _crack,
                 "edge_chip": _edge_chip, "flash": _flash}


def make_part(spec: ImpellerSpec, defect: str | None, rng: np.random.Generator,
              retries: int = 5) -> tuple[cq.Workplane, dict]:
    """Build one part. defect=None gives a good (OK) part.

    Returns the solid and metadata describing exactly what was changed:
    the ground truth for the generated image.
    """
    if defect == "bore_oversize":
        oversize = float(rng.uniform(1.2, 3.0))
        part = build_impeller(spec, bore_d=spec.bore_d + oversize)
        return part, {"defect": defect, "bore_d_actual": round(spec.bore_d + oversize, 2)}
    base = build_impeller(spec)
    if defect is None:
        return base, {"defect": "none"}
    if defect not in _DEFECT_FUNCS:
        raise ValueError(f"Unknown defect '{defect}'. Choose from {ALL_DEFECTS}")
    for _ in range(retries):  # boolean ops can occasionally fail on odd geometry
        try:
            part, info = _DEFECT_FUNCS[defect](base, spec, rng)
            if part.val().isValid():
                return part, {"defect": defect, **info}
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Could not create defect '{defect}' after {retries} tries")


def tessellate(part: cq.Workplane, tolerance: float = 0.08,
               angular: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """CAD solid (exact B-rep) -> triangle mesh (vertices Nx3, faces Mx3)."""
    verts, tris = part.val().tessellate(tolerance, angular)
    v = np.array([(p.x, p.y, p.z) for p in verts], dtype=np.float64)
    f = np.array(tris, dtype=np.int64)
    return v, f


def export_step(part: cq.Workplane, path: str) -> None:
    """Save the solid as a STEP file (opens in SolidWorks, CATIA, NX, FreeCAD...)."""
    cq.exporters.export(part, path)
