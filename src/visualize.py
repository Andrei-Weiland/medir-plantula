"""Desenho das marcacoes: caminho, 3 pontos-chave, segmentos e medidas."""
from __future__ import annotations

import cv2
import numpy as np

from .trace import Seedling

COLOR_SEG1 = (60, 200, 60)     # verde  (topo -> estrangulamento)
COLOR_SEG2 = (240, 160, 30)    # azul   (estrangulamento -> ponta)
COLOR_HEAD = (0, 215, 255)     # amarelo
COLOR_CONSTR = (0, 0, 255)     # vermelho
COLOR_TIP = (255, 0, 255)      # magenta


def _draw_polyline(img, pts, color, thickness):
    if len(pts) >= 2:
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], False, color, thickness, cv2.LINE_AA)


def draw_seedling(img: np.ndarray, s: Seedling, scale: float = 1.0) -> None:
    th = max(2, int(round(3 * scale)))
    r = max(4, int(round(7 * scale)))
    p1 = s.path[: s.constriction_idx + 1]
    p2 = s.path[s.constriction_idx:]
    _draw_polyline(img, p1, COLOR_SEG1, th)
    _draw_polyline(img, p2, COLOR_SEG2, th)

    cv2.circle(img, s.head, r, COLOR_HEAD, -1, cv2.LINE_AA)
    cv2.circle(img, s.constriction, r, COLOR_CONSTR, -1, cv2.LINE_AA)
    cv2.circle(img, s.tip, r, COLOR_TIP, -1, cv2.LINE_AA)

    label = f"{s.id + 1}"
    fs = 0.9 * scale
    lx, ly = s.head[0] + r + 2, s.head[1]
    cv2.putText(img, label, (lx + 1, ly + 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 0, 0), max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)


def annotate(
    bgr: np.ndarray,
    seedlings: list[Seedling],
    roi: tuple[int, int, int, int] | None = None,
    show_total_table: bool = True,
) -> np.ndarray:
    out = bgr.copy()
    scale = max(out.shape[:2]) / 1200.0
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (180, 180, 180),
                      max(1, int(2 * scale)))
    for s in seedlings:
        draw_seedling(out, s, scale)
    if show_total_table:
        _draw_table(out, seedlings, scale)
    return out


def _draw_table(img: np.ndarray, seedlings: list[Seedling], scale: float) -> None:
    if not seedlings:
        return
    fs = 0.6 * scale
    line_h = int(28 * scale)
    pad = int(12 * scale)
    rows = ["#  Seg1   Seg2   Total (cm)"] + [
        f"{s.id + 1:>2} {s.seg1_cm:6.2f} {s.seg2_cm:6.2f} {s.total_cm:6.2f}"
        for s in seedlings
    ]
    tw = int(max(len(r) for r in rows) * 11 * scale)
    th = line_h * len(rows) + 2 * pad
    x0, y0 = pad, pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + tw, y0 + th), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, row in enumerate(rows):
        y = y0 + pad + int((i + 0.8) * line_h)
        cv2.putText(img, row, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), max(1, int(1.4 * scale)), cv2.LINE_AA)
