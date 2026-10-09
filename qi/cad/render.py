"""Render CAD meshes into realistic top-view grayscale photos (synthetic data).

Pipeline (no GPU / OpenGL needed, pure NumPy + OpenCV):

1. rasterize_topdown(): z-buffer rasterization of the mesh seen from above ->
   a height map (mm) and a surface-normal map. Done ONCE per part.
2. shade(): turns height + normals into an image with
     - Lambert diffuse + weak specular light from a random direction
     - cast-iron surface roughness (bump-mapped noise texture)
     - ambient occlusion from the height map (cavities, cracks get darker)
     - shadows by marching the height map toward the light
3. camera(): random rotation, scale, shift, slight perspective tilt, blur,
   sensor noise and JPEG compression, then resized to 300x300 (same size as
   the real Kaggle images).

Randomising light, background, texture and camera for every image is called
DOMAIN RANDOMIZATION: the model sees so much variety in synthetic images that
real photos look like "just another variation", which reduces the
sim-to-real gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TopDownMaps:
    height: np.ndarray    # (R, R) float32, mm; NaN = background
    normals: np.ndarray   # (R, R, 3) float32 unit normals
    mm_per_px: float


def rasterize_topdown(verts: np.ndarray, faces: np.ndarray, extent_mm: float,
                      res: int = 600) -> TopDownMaps:
    """Orthographic top-down z-buffer. extent_mm = half-width of the view."""
    scale = res / (2 * extent_mm)
    px = (verts[:, 0] + extent_mm) * scale
    py = (extent_mm - verts[:, 1]) * scale
    pz = verts[:, 2]

    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)
    visible = n[:, 2] > 0.02               # only faces that point up are seen from above

    H = np.full((res, res), -np.inf, dtype=np.float64)
    N = np.zeros((res, res, 3), dtype=np.float32)
    for fi in np.nonzero(visible)[0]:
        i0, i1, i2 = faces[fi]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        xmin, xmax = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), res - 1))
        ymin, ymax = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), res - 1))
        if xmax < xmin or ymax < ymin:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xs = np.arange(xmin, xmax + 1) + 0.5
        ys = np.arange(ymin, ymax + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
        w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        z = w0 * pz[i0] + w1 * pz[i1] + w2 * pz[i2]
        sub = H[ymin:ymax + 1, xmin:xmax + 1]
        upd = inside & (z > sub)
        sub[upd] = z[upd]
        N[ymin:ymax + 1, xmin:xmax + 1][upd] = n[fi]

    # Fill pinholes left by sliver triangles (pixels between covered pixels).
    mask = np.isfinite(H).astype(np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    holes = (closed == 1) & (mask == 0)
    if holes.any():
        Hf = np.where(np.isfinite(H), H, -1e9).astype(np.float32)
        Hd = cv2.dilate(Hf, np.ones((3, 3), np.uint8))
        H[holes] = Hd[holes]
        for c in range(3):
            Nd = cv2.dilate(N[:, :, c], np.ones((3, 3), np.uint8))
            N[:, :, c][holes] = Nd[holes]
    height = np.where(np.isfinite(H), H, np.nan).astype(np.float32)
    N[~np.isfinite(H)] = (0, 0, 1)
    return TopDownMaps(height=height, normals=N, mm_per_px=1 / scale)


def _fractal_noise(shape, rng, octaves=(2, 4, 8, 16)):
    out = np.zeros(shape, np.float32)
    for i, s in enumerate(octaves):
        small = rng.standard_normal((max(shape[0] // s, 2), max(shape[1] // s, 2))).astype(np.float32)
        out += cv2.resize(small, shape[::-1], interpolation=cv2.INTER_CUBIC) / (2 ** i)
    return out / (np.abs(out).max() + 1e-6)


def shade(maps: TopDownMaps, rng: np.random.Generator) -> np.ndarray:
    """Height + normals -> float image 0..1 with random, realistic lighting."""
    H, N = maps.height, maps.normals.copy()
    part = np.isfinite(H)
    h = np.where(part, H, 0).astype(np.float32)
    shape = H.shape

    # Cast-surface roughness: perturb normals with a fine noise bump map.
    rough = rng.uniform(0.15, 0.45)
    bump = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), rng.uniform(0.8, 1.6))
    gx, gy = cv2.Sobel(bump, cv2.CV_32F, 1, 0), cv2.Sobel(bump, cv2.CV_32F, 0, 1)
    N[..., 0] -= rough * gx
    N[..., 1] += rough * gy
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-6

    az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(35, 80))
    L = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], np.float32)
    lambert = np.clip(N @ L, 0, 1)
    half = L + np.array([0, 0, 1], np.float32)
    half /= np.linalg.norm(half)
    spec = np.clip(N @ half, 0, 1) ** rng.uniform(12, 40) * rng.uniform(0.05, 0.25)

    # Ambient occlusion: points lower than their neighbourhood are darker.
    ao = np.zeros(shape, np.float32)
    for sigma in (1.5, 3, 6):
        ao += np.clip(cv2.GaussianBlur(h, (0, 0), sigma) - h, 0, None)
    ao = np.exp(-ao * rng.uniform(0.5, 0.8))

    # Shadows: march toward the light; blocked if terrain rises above the ray.
    shadow = np.ones(shape, np.float32)
    px_per_mm = 1 / maps.mm_per_px
    dx, dy = np.cos(az), -np.sin(az)
    rise_per_px = np.tan(el) / px_per_mm
    for k in range(1, 40, 2):
        M = np.float32([[1, 0, -dx * k], [0, 1, -dy * k]])
        shifted = cv2.warpAffine(h, M, shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0)
        shadow = np.minimum(shadow, np.where(shifted > h + k * rise_per_px + 0.05, 0.45, 1.0))
    shadow = cv2.GaussianBlur(shadow, (0, 0), 1.2)

    albedo = rng.uniform(0.45, 0.7) * (1 + 0.12 * _fractal_noise(shape, rng))
    ambient = rng.uniform(0.18, 0.32)
    img = albedo * (ambient * ao + (1 - ambient) * lambert * shadow * ao) + spec * shadow

    # Background: a table/conveyor with gradient and texture.
    bg_level = rng.uniform(0.05, 0.75)
    gy_, gx_ = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32) / shape[0]
    grad = (gx_ * np.cos(az) + gy_ * np.sin(az)) * rng.uniform(-0.15, 0.15)
    bg = bg_level + grad + 0.04 * _fractal_noise(shape, rng)
    # soft contact shadow of the part on the background
    cast = cv2.GaussianBlur(part.astype(np.float32), (0, 0), 6)
    bg = bg * (1 - 0.35 * cast)
    # Background seen through deep holes (the shaft bore) gets little light:
    # same occlusion idea as above, using the floor (height 0) inside the hole.
    hole_ao = np.zeros(shape, np.float32)
    for sigma in (3, 8):
        hole_ao += np.clip(cv2.GaussianBlur(h, (0, 0), sigma), 0, None)
    bg = bg * np.exp(-hole_ao * rng.uniform(0.06, 0.12))
    return np.clip(np.where(part, img, bg), 0, 1).astype(np.float32)


def pose(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random in-plane rotation, scale and position of the part under the camera.

    Applied to the 3D model itself (not the image), so no image borders appear.
    """
    a = np.radians(rng.uniform(0, 360))
    s = rng.uniform(0.92, 1.06)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    v = verts.copy()
    v[:, :2] = (v[:, :2] @ R.T) * s + rng.uniform(-3, 3, 2)
    v[:, 2] *= s
    return v


def camera(img: np.ndarray, rng: np.random.Generator, out_size: int = 300) -> np.ndarray:
    """Optics + sensor effects. Returns uint8 grayscale."""
    h, w = img.shape
    # slight perspective tilt (camera not perfectly above the part)
    t = rng.uniform(0, 0.02) * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-t, t, (4, 2)).astype(np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REFLECT)
    out = cv2.resize(out, (out_size, out_size), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.1))
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)   # exposure
    out = out + rng.normal(0, rng.uniform(0.005, 0.025), out.shape)  # sensor noise
    out8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def render_views(verts: np.ndarray, faces: np.ndarray, outer_r: float, n_views: int,
                 rng: np.random.Generator, out_size: int = 300, res: int = 600,
                 fill: float = 0.86) -> list[np.ndarray]:
    """Many different-looking photos of the same physical part.

    fill = fraction of the image width the part diameter covers (the real
    dataset images are close-ups, so the part nearly fills the frame).
    """
    extent = outer_r / fill
    views = []
    for _ in range(n_views):
        maps = rasterize_topdown(pose(verts, rng), faces, extent_mm=extent, res=res)
        views.append(camera(shade(maps, rng), rng, out_size))
    return views
