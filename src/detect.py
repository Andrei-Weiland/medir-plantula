"""Deteccao automatica das plantulas.

Estrategia:
  - "cabecas" = cotiledones amarelos e sementes escuras (topo da estrutura branca);
  - de cada cabeca, segue-se a estrutura ate o ponto geodesicamente mais distante
    (a ponta da raiz), usando o mesmo mapa de custo do tracado;
  - plantulas duplicadas (caminhos sobrepostos) sao fundidas.

As raizes podem se cruzar/encostar, entao a deteccao automatica pode falhar em
alguns casos -> a camada interativa permite corrigir (modo hibrido).
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from .structure import Structures
from .trace import Seedling, build_seedling


def _head_blobs(bgr: np.ndarray, structures: Structures) -> list[tuple[int, int]]:
    """Centroides de cotiledones (amarelo) e sementes (escuro) dentro da ROI."""
    roi_mask = (structures.cost < 1e5).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    yellow = ((H >= 18) & (H <= 45) & (S > 60) & (V > 80)).astype(np.uint8)
    dark = (V < 90).astype(np.uint8)
    heads = ((yellow | dark) & roi_mask).astype(np.uint8) * 255
    heads = cv2.morphologyEx(heads, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    heads = cv2.morphologyEx(heads, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    n, _, stats, cents = cv2.connectedComponentsWithStats(heads)
    out = []
    img_area = bgr.shape[0] * bgr.shape[1]
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < img_area * 2e-5:  # remove ruido
            continue
        if area > img_area * 0.01:  # remove manchas enormes
            continue
        out.append((int(cents[i][0]), int(cents[i][1])))
    return out


def _snap_to_structure(
    point: tuple[int, int], mask: np.ndarray, radius: int = 60
) -> tuple[int, int] | None:
    """Move o ponto para o pixel de estrutura mais proximo (dentro de `radius`)."""
    x, y = point
    h, w = mask.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return None
    d = (xs - (x - x0)) ** 2 + (ys - (y - y0)) ** 2
    j = int(np.argmin(d))
    return (int(xs[j] + x0), int(ys[j] + y0))


def farthest_tip(
    structures: Structures, head_xy: tuple[int, int]
) -> tuple[int, int] | None:
    """Ponto geodesicamente mais distante da cabeca dentro do componente da raiz."""
    mask = structures.mask
    num, labels = cv2.connectedComponents(mask)
    comp = labels[head_xy[1], head_xy[0]]
    if comp == 0:
        snapped = _snap_to_structure(head_xy, mask)
        if snapped is None:
            return None
        head_xy = snapped
        comp = labels[head_xy[1], head_xy[0]]
        if comp == 0:
            return None

    comp_mask = labels == comp
    # custo: dentro do componente usa o custo normal, fora e proibido
    cost = structures.cost.copy()
    cost[~comp_mask] = 1e6

    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs([(head_xy[1], head_xy[0])])
    cum = np.where(comp_mask, cum, np.inf)
    if not np.isfinite(cum).any():
        return None
    ridx = np.unravel_index(np.argmax(np.where(np.isfinite(cum), cum, -1)), cum.shape)
    return (int(ridx[1]), int(ridx[0]))


def detect_seedlings(
    bgr: np.ndarray, structures: Structures, px_per_cm: float
) -> list[Seedling]:
    heads = _head_blobs(bgr, structures)
    seedlings: list[Seedling] = []
    used_tips: list[tuple[int, int]] = []

    sid = 0
    for hx, hy in heads:
        snapped = _snap_to_structure((hx, hy), structures.mask)
        if snapped is None:
            continue
        tip = farthest_tip(structures, snapped)
        if tip is None:
            continue
        # evitar duplicatas: pontas muito proximas de uma ja usada
        if any((tip[0] - tx) ** 2 + (tip[1] - ty) ** 2 < 60 ** 2
               for tx, ty in used_tips):
            continue
        try:
            s = build_seedling(
                sid, structures.cost, structures.width, snapped, tip, px_per_cm
            )
        except Exception:
            continue
        # descartar caminhos curtos demais (ruido)
        if s.total_px < 80:
            continue
        seedlings.append(s)
        used_tips.append(tip)
        sid += 1

    return seedlings
