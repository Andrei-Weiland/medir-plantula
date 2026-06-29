"""Exportacao dos resultados: tabela CSV."""
from __future__ import annotations

import csv
from pathlib import Path

from .trace import Seedling


def save_csv(seedlings: list[Seedling], path: str | Path) -> None:
    """Grava tabela CSV com medidas de cada plantula."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "plantula", "seg1_topo_estrangulamento_cm",
            "seg2_estrangulamento_raiz_cm", "total_cm",
            "seg1_px", "seg2_px", "total_px", "origem",
        ])
        for s in seedlings:
            w.writerow([
                s.id + 1,
                f"{s.seg1_cm:.2f}", f"{s.seg2_cm:.2f}", f"{s.total_cm:.2f}",
                f"{s.seg1_px:.1f}", f"{s.seg2_px:.1f}", f"{s.total_px:.1f}",
                "manual" if s.manual else "auto",
            ])
