from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.graph import route_through_array


@dataclass
class Seedling:
    id: int
    head: tuple[int, int]
    constriction: tuple[int, int]
    tip: tuple[int, int]
    path: list[tuple[int, int]]
    constriction_idx: int
    px_per_cm: float = 0.0
    seg1_px: float = 0.0
    seg2_px: float = 0.0
    manual: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def seg1_cm(self) -> float:
        return self.seg1_px / self.px_per_cm if self.px_per_cm else 0.0

    @property
    def seg2_cm(self) -> float:
        return self.seg2_px / self.px_per_cm if self.px_per_cm else 0.0

    @property
    def total_cm(self) -> float:
        return self.seg1_cm + self.seg2_cm

    @property
    def total_px(self) -> float:
        return self.seg1_px + self.seg2_px


def trace_path(
    cost: np.ndarray, start_xy: tuple[int, int], end_xy: tuple[int, int]
) -> list[tuple[int, int]]:
    """Caminho de menor custo entre dois pontos (x, y). Retorna lista de (x, y)."""
    h, w = cost.shape
    sx = int(np.clip(start_xy[0], 0, w - 1))
    sy = int(np.clip(start_xy[1], 0, h - 1))
    ex = int(np.clip(end_xy[0], 0, w - 1))
    ey = int(np.clip(end_xy[1], 0, h - 1))
    indices, _ = route_through_array(
        cost, (sy, sx), (ey, ex), fully_connected=True, geometric=True
    )
    return [(int(c), int(r)) for r, c in indices]


def smooth_path(
    path: list[tuple[int, int]], window: int = 9
) -> np.ndarray:
    """Suaviza o caminho (media movel) para medir comprimento sem escada de pixels."""
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) < window or window < 3:
        return pts
    k = window if window % 2 == 1 else window + 1
    pad = k // 2
    padded = np.vstack(
        [np.repeat(pts[:1], pad, axis=0), pts, np.repeat(pts[-1:], pad, axis=0)]
    )
    kernel = np.ones(k) / k
    xs = np.convolve(padded[:, 0], kernel, mode="valid")
    ys = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.column_stack([xs, ys])


def path_length_px(path: list[tuple[int, int]], smooth: bool = True) -> float:
    """Comprimento curvilineo (px) somando distancias entre pontos consecutivos."""
    pts = smooth_path(path) if smooth else np.asarray(path, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def width_profile(
    path: list[tuple[int, int]], width_map: np.ndarray
) -> np.ndarray:
    """Largura (raio em px, via distance transform) ao longo do caminho."""
    h, w = width_map.shape
    vals = []
    for x, y in path:
        xi = min(max(int(x), 0), w - 1)
        yi = min(max(int(y), 0), h - 1)
        vals.append(width_map[yi, xi])
    return np.asarray(vals, dtype=np.float64)


def _smooth1d(a: np.ndarray, k: int = 15) -> np.ndarray:
    if len(a) < k or k < 3:
        return a
    pad = k // 2
    padded = np.concatenate([np.repeat(a[:1], pad), a, np.repeat(a[-1:], pad)])
    return np.convolve(padded, np.ones(k) / k, mode="valid")[: len(a)]


def find_constriction_idx(
    path: list[tuple[int, int]], width_map: np.ndarray
) -> int:
    """Estima o indice do ponto de estrangulamento ao longo do caminho."""
    n = len(path)
    if n < 8:
        return n // 2

    prof = width_profile(path, width_map)
    prof = _smooth1d(prof, k=max(5, n // 30 | 1))

    lo = max(2, int(0.05 * n))
    hi = max(lo + 2, int(0.55 * n))

    grad = np.gradient(prof)
    idx = lo + int(np.argmin(grad[lo:hi]))
    return int(np.clip(idx, 1, n - 2))


def build_seedling(
    sid: int,
    cost: np.ndarray,
    width_map: np.ndarray,
    head_xy: tuple[int, int],
    tip_xy: tuple[int, int],
    px_per_cm: float,
    constriction_xy: tuple[int, int] | None = None,
    manual: bool = False,
) -> Seedling:
    """Tra\u00e7a o caminho topo->ponta e calcula os dois segmentos."""
    path = trace_path(cost, head_xy, tip_xy)
    if len(path) < 2:
        path = [tuple(map(int, head_xy)), tuple(map(int, tip_xy))]

    if constriction_xy is not None:
        pts = np.asarray(path)
        d = ((pts[:, 0] - constriction_xy[0]) ** 2
             + (pts[:, 1] - constriction_xy[1]) ** 2)
        c_idx = int(np.argmin(d))
    else:
        c_idx = find_constriction_idx(path, width_map)

    c_idx = int(np.clip(c_idx, 1, len(path) - 2))
    seg1 = path_length_px(path[: c_idx + 1])
    seg2 = path_length_px(path[c_idx:])

    return Seedling(
        id=sid,
        head=tuple(map(int, path[0])),
        constriction=tuple(map(int, path[c_idx])),
        tip=tuple(map(int, path[-1])),
        path=path,
        constriction_idx=c_idx,
        px_per_cm=px_per_cm,
        seg1_px=seg1,
        seg2_px=seg2,
        manual=manual,
    )
