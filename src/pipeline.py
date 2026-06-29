from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import structure as st
from . import detect as dt
from .trace import Seedling


@dataclass
class Work:
    bgr: np.ndarray
    scale: float
    roi: tuple[int, int, int, int]
    roi_mask: np.ndarray
    structures: st.Structures


def prepare(bgr_orig: np.ndarray, max_dim: int = 1400) -> Work:
    """Reduz a imagem para a resolucao de trabalho e extrai as estruturas."""
    h, w = bgr_orig.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr_orig, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    else:
        bgr = bgr_orig.copy()
    roi, roi_mask = st.find_paper(bgr)
    structures = st.extract(bgr, roi, roi_mask)
    return Work(bgr=bgr, scale=scale, roi=roi, roi_mask=roi_mask,
                structures=structures)


def auto_detect(work: Work, px_per_cm: float) -> list[Seedling]:
    """Deteccao automatica de plantulas na resolucao de trabalho."""
    return dt.detect_seedlings(work.bgr, work.structures, px_per_cm)
