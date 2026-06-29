from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Calibration:
    px_per_cm: float
    method: str
    confidence: float = 1.0
    detail: str = ""


def from_two_points(
    p1: tuple[float, float], p2: tuple[float, float], real_cm: float
) -> Calibration:
    """px_per_cm a partir de dois pontos e a distancia real entre eles (cm)."""
    if real_cm <= 0:
        raise ValueError("A distancia real deve ser > 0 cm.")
    dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    return Calibration(px_per_cm=dist / real_cm, method="cliques",
                       detail=f"{dist:.1f}px = {real_cm}cm")


def _cm_ticks_in_band(
    gray: np.ndarray, y0: int, y1: int, W: int
) -> tuple[float, float, int] | None:
    """Detecta marcas de cm (ticks mais altos) numa faixa. Retorna
    (px_per_cm, consistencia, n_ticks) ou None."""
    from scipy.signal import find_peaks

    band = gray[y0:y1, :]
    if band.size == 0:
        return None
    dark = (band < band.mean() - 0.4 * band.std()).astype(np.uint8) * 255
    kh = max(21, int((y1 - y0) * 0.5) | 1)
    tall = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))
    col = tall.sum(axis=0).astype(np.float32)
    if col.max() <= 0:
        return None
    col /= col.max()
    peaks, _ = find_peaks(col, height=0.3, distance=int(W * 0.025))
    if len(peaks) < 4:
        return None
    diffs = np.diff(peaks).astype(np.float64)
    med = float(np.median(diffs))
    if med <= 0:
        return None
    inliers = diffs[np.abs(diffs - med) < 0.18 * med]
    if len(inliers) < 3:
        return None
    consistency = len(inliers) / len(diffs)
    return float(np.median(inliers)), consistency, len(peaks)


def from_ruler(bgr: np.ndarray) -> Calibration | None:
    """Detecta px por cm pelas marcas de cm da regua (varre o topo da imagem).

    Mantem apenas os ticks mais altos (marcas de cm) para evitar confundir com
    marcas de mm/5mm. Retorna None se nao for consistente -> usar cliques.
    """
    try:
        import scipy.signal  # noqa: F401
    except Exception:
        return None

    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    win = max(60, int(H * 0.035))

    cands = []
    for y0 in range(int(H * 0.04), int(H * 0.22), max(20, win // 4)):
        res = _cm_ticks_in_band(gray, y0, y0 + win, W)
        if res is None:
            continue
        ppc, cons, n = res
        if cons >= 0.7 and n >= 6:
            cands.append((ppc, cons, n, y0))

    if not cands:
        return None

    cands.sort(key=lambda c: c[0], reverse=True)
    ppc, cons, n, y0 = cands[0]
    return Calibration(
        px_per_cm=ppc, method="regua", confidence=round(cons, 2),
        detail=f"{n} marcas de cm em y~{y0}, ~{ppc:.1f}px/cm",
    )
