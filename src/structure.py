"""Extracao das estruturas da plantula (filamentos brancos sobre papel branco).

A ideia central: filamentos brancos e raizes tem baixissimo contraste contra o
papel branco num threshold direto, mas aparecem nitidamente ao subtrair um fundo
borrado (a iluminacao/papel variam devagar; os filamentos sao detalhes finos).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Structures:
    roi: tuple[int, int, int, int]  # x, y, w, h da caixa (regiao util)
    detail: np.ndarray              # realce dos filamentos (float32 0..1)
    prob: np.ndarray                # probabilidade de ser estrutura (float32 0..1)
    mask: np.ndarray                # binario uint8 (0/255) das estruturas
    width: np.ndarray               # distance transform (raio em px) dentro da mascara
    cost: np.ndarray                # custo para caminho de menor custo (float32)


def find_paper(bgr: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Encontra o quadrado de papel branco (substrato) onde estao as plantulas.

    O papel e a maior regiao branca *compacta* (alto brilho, baixa saturacao).
    A regua (no topo) fica conectada ao papel por uma ponte fina (a borda da
    caixa); uma erosao vertical corta essa ponte para isolar so o papel.

    Retorna (bbox, mask) onde mask e o poligono do papel preenchido (uint8 0/255).
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    paper = ((v > 165) & (s < 45)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    # cortar a ponte fina (regua <-> papel) com erosao vertical
    cut = cv2.erode(paper, np.ones((31, 5), np.uint8))

    cnts, _ = cv2.findContours(cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        full = np.ones((H, W), np.uint8) * 255
        return (0, 0, W, H), full

    biggest = max(cnts, key=cv2.contourArea)
    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((31, 5), np.uint8))

    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, w, h), mask


def find_paper_roi(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Compatibilidade: retorna apenas o bbox do papel."""
    return find_paper(bgr)[0]


def _background_subtract(gray: np.ndarray, sigma: float) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    detail = cv2.subtract(gray, bg)
    detail = detail.astype(np.float32)
    m = detail.max()
    if m > 0:
        detail /= m
    return detail


def extract(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    roi_mask: np.ndarray | None = None,
    sigma: float | None = None,
    prob_thresh: float = 0.12,
) -> Structures:
    """Constroi mascara/probabilidade/custo das estruturas dentro da ROI."""
    if roi is None or roi_mask is None:
        roi, roi_mask = find_paper(bgr)
    x, y, w, h = roi

    # escala do realce proporcional ao tamanho do papel (filamentos finos)
    if sigma is None:
        sigma = max(5.0, w * 0.0085)

    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detail_full = _background_subtract(gray_full, sigma)

    # restringir tudo a ROI (fora dela, custo alto / prob zero)
    roi_mask = (roi_mask > 0).astype(np.uint8)

    detail = detail_full * roi_mask

    # cotiledones amarelos e sementes tambem fazem parte da estrutura (cabeca)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((H >= 20) & (H <= 45) & (S > 60) & (V > 80)).astype(np.float32)
    dark = (V < 95).astype(np.float32)
    head = np.clip(yellow + dark, 0, 1) * roi_mask

    prob = np.clip(detail * 1.6 + head, 0, 1).astype(np.float32)
    prob = cv2.GaussianBlur(prob, (0, 0), 1.0)

    mask = (prob > prob_thresh).astype(np.uint8) * 255
    # remover ruido isolado
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # conectar lacunas AO LONGO do filamento (sao ~verticais) sem fundir
    # plantulas vizinhas (separadas horizontalmente)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vkernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    width = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    # custo para route_through_array: baixo onde prob alta; muito alto fora da ROI
    cost = (1.0 - prob).astype(np.float32)
    cost = cost * cost * 50.0 + 0.05
    cost[roi_mask == 0] = 1e6

    return Structures(
        roi=roi, detail=detail, prob=prob, mask=mask, width=width, cost=cost
    )
