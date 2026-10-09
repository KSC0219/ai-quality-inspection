"""Dimensional inspection: measure the shaft bore in a photo and compare it with
the CAD nominal value and tolerance.

How it works
1. Find the part's outer edge (largest circle) with the Hough circle transform.
2. Find the hub circle. Scale: we know the CAD hub diameter (30 mm), so
       mm_per_px = hub_d_mm / hub_d_px
   The part itself acts as the ruler, so no separate camera calibration is needed.
3. Find the bore (small circle inside the hub, size predicted from the CAD ratio).
   Each circle is fitted to ~100 edge points with least squares, ignoring
   outliers, which gives sub-pixel accuracy.
4. bore_mm = bore_px * mm_per_px  ->  PASS if |bore_mm - nominal| <= tolerance.

This is the "check against the CAD drawing" step of real inspection systems.
Limits: assumes a top-down view; accuracy is about +/-0.3 mm at 300x300 px
(about 0.4 mm per pixel). A higher-resolution camera gives proportionally better accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np

from .part import ImpellerSpec


def _to_gray(image) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)


def find_outer_circle(gray: np.ndarray) -> tuple[float, float, float] | None:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    for p2 in (60, 45, 30, 20):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w,
                             param1=60, param2=p2, minRadius=int(0.30 * w), maxRadius=int(0.52 * w))
        if c is not None:
            x, y, r = c[0][0]
            return float(x), float(y), float(r)
    return None


def find_bore(gray: np.ndarray, center: tuple[float, float], expected_r: float) -> tuple[float, float, float] | None:
    cx, cy = center
    win = int(expected_r * 3)
    x0, y0 = max(int(cx - win), 0), max(int(cy - win), 0)
    crop = gray[y0:int(cy + win), x0:int(cx + win)]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)   # sub-pixel accuracy
    blur = cv2.GaussianBlur(crop, (0, 0), 2.5)
    er = expected_r * 3
    best = None
    for p2 in (40, 30, 22, 15):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=er / 2, param1=60, param2=p2,
                             minRadius=int(er * 0.6), maxRadius=int(er * 1.5))
        if c is not None:
            ccx, ccy = crop.shape[1] / 2, crop.shape[0] / 2
            best = min(c[0], key=lambda k: (k[0] - ccx) ** 2 + (k[1] - ccy) ** 2)
            break
    if best is None:
        return None
    bx, by, br = best
    return x0 + bx / 3, y0 + by / 3, br / 3


def _fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
    """Least-squares (Kasa) circle fit: x^2+y^2 + Dx + Ey + F = 0."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    return cx, cy, float(np.sqrt(max(cx ** 2 + cy ** 2 - F, 0)))


def outline_from_texture(gray: np.ndarray) -> tuple[float, float, float] | None:
    """Part outline from surface texture: a cast surface is rough, the table is
    smooth. Works even when part and background have the same brightness."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (7, 7))
    std = np.sqrt(np.clip(cv2.blur(g * g, (7, 7)) - mean * mean, 0, None))
    std = cv2.GaussianBlur(std, (0, 0), 2)
    s8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < 30:
        return None
    for _ in range(3):
        fx, fy, fr = _fit_circle(c)
        res = np.abs(np.hypot(c[:, 0] - fx, c[:, 1] - fy) - fr)
        c = c[res < max(1.5, 2.5 * np.median(res))]
    fx, fy, fr = _fit_circle(c)
    h, w = gray.shape
    return (fx, fy, fr) if 0.2 * w < fr < 0.6 * w else None


def refine_circle(gray: np.ndarray, center, r_guess: float, lo: float = 0.7, hi: float = 1.3,
                  n_rays: int = 120, return_strength: bool = False):
    """Sub-pixel circle: on each ray from the centre, take the strongest intensity
    edge between lo*r and hi*r, then fit a circle and drop outliers (vanes, defects)."""
    img = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    cx, cy = center
    radii = np.arange(lo * r_guess, hi * r_guess, 0.2, dtype=np.float32)
    if radii.size < 5:
        return None
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    X = (cx + np.cos(ang)[:, None] * radii[None, :]).astype(np.float32)
    Y = (cy + np.sin(ang)[:, None] * radii[None, :]).astype(np.float32)
    prof = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    grad = np.abs(np.gradient(prof, axis=1))
    k = grad.argmax(axis=1)
    strength = grad[np.arange(n_rays), k]
    keep = strength > np.percentile(strength, 30)          # skip weak / ambiguous rays
    pts = np.column_stack([X[np.arange(n_rays), k], Y[np.arange(n_rays), k]])[keep]
    if len(pts) < 12:
        return None
    for _ in range(3):                                      # robust: refit without outliers
        fx, fy, fr = _fit_circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - fx, pts[:, 1] - fy) - fr)
        good = res < max(1.0, 2.5 * np.median(res))
        if good.sum() < 12:
            break
        pts = pts[good]
    circle = _fit_circle(pts)
    if return_strength:
        return circle, float(np.median(strength[keep]))
    return circle


def _plain(obj):
    """NumPy numbers -> plain Python numbers (so results can be sent as JSON)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def measure_bore(image, spec: ImpellerSpec = ImpellerSpec()) -> dict:
    """Returns measured bore diameter (mm), deviation from CAD nominal and PASS/FAIL."""
    return _plain(_measure_bore(image, spec))


def _measure_bore(image, spec: ImpellerSpec) -> dict:
    gray = _to_gray(image)
    outer = find_outer_circle(gray)
    if outer is None:
        return {"status": "NOT_FOUND", "reason": "part outline not detected"}
    # 1. rim: sharpen on intensity edges in a band that excludes the vane tips
    ox, oy, orad = refine_circle(gray, outer[:2], outer[2], 0.94, 1.06) or outer
    # 2. hub = the ruler. It sits right next to the bore (same height, no
    #    perspective difference) and its nominal size is on the CAD drawing.
    hub = refine_circle(gray, (ox, oy), orad * spec.hub_d / spec.outer_d, 0.8, 1.25)
    if hub is None:
        return {"status": "NOT_FOUND", "reason": "hub not detected"}
    mm_per_px = spec.hub_d / (2 * hub[2])
    # 3. bore: between 0.6x and 1.5x its expected size, so the hub edge is excluded
    fit = refine_circle(gray, hub[:2], hub[2] * spec.bore_d / spec.hub_d, 0.6, 1.5,
                        n_rays=90, return_strength=True)
    if fit is None:
        return {"status": "NOT_FOUND", "reason": "bore not detected", "mm_per_px": round(mm_per_px, 4)}
    bore, edge_strength = fit
    bore_mm = 2 * bore[2] * mm_per_px
    dev = bore_mm - spec.bore_d
    if edge_strength < 3.0:      # bore edge barely visible: don't guess, ask for better light
        status = "UNCERTAIN"
    else:
        status = "PASS" if abs(dev) <= spec.bore_tol else "FAIL"
    return {
        "status": status, "edge_contrast": round(edge_strength, 1),
        "bore_mm": round(bore_mm, 2), "nominal_mm": spec.bore_d, "tolerance_mm": spec.bore_tol,
        "deviation_mm": round(dev, 2), "mm_per_px": round(mm_per_px, 4),
        "outer_circle_px": [round(ox, 1), round(oy, 1), round(orad, 1)],
        "hub_circle_px": [round(hub[0], 1), round(hub[1], 1), round(hub[2], 1)],
        "bore_circle_px": [round(bore[0], 1), round(bore[1], 1), round(bore[2], 1)],
    }


def draw_measurement(image, result: dict) -> np.ndarray:
    """Overlay detected circles and the verdict on the image (RGB uint8)."""
    gray = _to_gray(image)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if "outer_circle_px" in result:
        x, y, r = result["outer_circle_px"]
        cv2.circle(out, (int(x), int(y)), int(r), (60, 140, 255), 1)
    if "bore_circle_px" in result:
        x, y, r = result["bore_circle_px"]
        color = (40, 200, 90) if result["status"] == "PASS" else (230, 60, 50)
        cv2.circle(out, (int(x), int(y)), max(int(round(r)), 1), color, 2)
        cv2.putText(out, f"bore {result['bore_mm']} mm {result['status']}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
