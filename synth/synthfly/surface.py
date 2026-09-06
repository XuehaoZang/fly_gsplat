"""Part-labelled surface samples on the fly's visual meshes.

Samples are drawn once, area-weighted, on the mesh triangles of every visible
mesh geom and stored in the geom's local frame. Per frame they are moved with
the geom (rigid), which gives temporal correspondence for free: sample i is
the same material point in every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import PART_IDS, PART_NAMES


@dataclass
class SurfaceSamples:
    local: np.ndarray  # (P, 3) in geom frame, cm
    geom_id: np.ndarray  # (P,) int
    part_id: np.ndarray  # (P,) uint8
    material_gray: np.ndarray  # (P,) uint8, expected backlit intensity of the material
    normal_local: np.ndarray  # (P, 3) triangle normal in geom frame

    @property
    def count(self) -> int:
        return int(self.local.shape[0])


def _triangles_of_geom(model, gid: int):
    mid = int(model.geom_dataid[gid])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64)
    faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int64)
    return verts[faces]  # (F, 3, 3)


def _material_gray(model, gid: int) -> int:
    """Backlit intensity of an opaque-black material with alpha a over white: 255 * (1 - a)."""
    mat = int(model.geom_matid[gid])
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[gid]
    return int(round(255.0 * (1.0 - float(rgba[3]))))


def sample_surface(scene, counts: dict[str, int], seed: int = 0) -> SurfaceSamples:
    """Draw `counts[part]` samples per part on the scene's visible mesh geoms."""
    model = scene.model
    rng = np.random.default_rng(seed)
    locals_, geoms, parts, grays, normals = [], [], [], [], []
    for part, n in counts.items():
        if part not in PART_IDS or part == "background":
            raise KeyError(f"unknown part {part!r}; known: {PART_NAMES[1:]}")
        n = int(n)
        if n <= 0:
            continue
        pid = PART_IDS[part]
        gids = [int(g) for g in scene.visual_geoms if scene.geom_part[g] == pid]
        if not gids:
            continue
        tris, tri_geom = [], []
        for g in gids:
            t = _triangles_of_geom(model, g)
            tris.append(t)
            tri_geom.append(np.full(len(t), g, dtype=np.int64))
        tris = np.concatenate(tris)
        tri_geom = np.concatenate(tri_geom)
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        area = 0.5 * np.linalg.norm(cross, axis=1)
        if area.sum() <= 0:
            continue
        pick = rng.choice(len(tris), size=n, p=area / area.sum())
        r1 = np.sqrt(rng.random(n))
        r2 = rng.random(n)
        a, b, c = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
        pts = a[:, None] * tris[pick, 0] + b[:, None] * tris[pick, 1] + c[:, None] * tris[pick, 2]
        nrm = cross[pick] / np.maximum(np.linalg.norm(cross[pick], axis=1, keepdims=True), 1e-12)
        locals_.append(pts)
        geoms.append(tri_geom[pick])
        parts.append(np.full(n, pid, dtype=np.uint8))
        grays.append(np.array([_material_gray(model, g) for g in tri_geom[pick]], dtype=np.uint8))
        normals.append(nrm)
    if not locals_:
        raise ValueError("no surface samples requested")
    return SurfaceSamples(
        local=np.concatenate(locals_),
        geom_id=np.concatenate(geoms),
        part_id=np.concatenate(parts),
        material_gray=np.concatenate(grays),
        normal_local=np.concatenate(normals),
    )


def world_points(samples: SurfaceSamples, geom_xpos: np.ndarray, geom_xmat: np.ndarray) -> np.ndarray:
    """(P, 3) world positions of the samples for the current geom transforms."""
    R = geom_xmat[samples.geom_id]  # (P, 3, 3)
    p = geom_xpos[samples.geom_id]  # (P, 3)
    return np.einsum("pij,pj->pi", R, samples.local) + p


def world_normals(samples: SurfaceSamples, geom_xmat: np.ndarray) -> np.ndarray:
    return np.einsum("pij,pj->pi", geom_xmat[samples.geom_id], samples.normal_local)


def visibility(points_world: np.ndarray, calibration, depth_img: np.ndarray, tol_cm: float):
    """Which samples are visible in one camera: inside the image and not behind the rendered surface.

    Returns (visible (P,) bool, uv (P, 2) float, pixel gray sampling indices (P, 2) int)."""
    uv, depth = calibration.project(points_world)
    H, W = depth_img.shape
    col = np.floor(uv[:, 0]).astype(np.int64)
    row = np.floor(uv[:, 1]).astype(np.int64)
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H) & (depth > 0)
    vis = np.zeros(len(uv), dtype=bool)
    ci, ri = col[inside], row[inside]
    surface = depth_img[ri, ci].astype(np.float64)
    vis[inside] = depth[inside] <= surface + tol_cm
    return vis, uv, np.stack([row, col], axis=1)
