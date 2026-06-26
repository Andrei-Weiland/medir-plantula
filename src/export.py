"""Exportacao dos resultados: tabela CSV e persistencia do estado (JSON)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .trace import Seedling


def save_csv(seedlings: list[Seedling], path: str | Path) -> None:
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


def save_state(
    seedlings: list[Seedling],
    px_per_cm: float,
    scale: float,
    path: str | Path,
) -> None:
    """Salva o estado para reabrir/editar depois (caminhos em coords de trabalho)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "px_per_cm": px_per_cm,
        "scale": scale,
        "seedlings": [
            {
                "id": s.id,
                "head": list(s.head),
                "tip": list(s.tip),
                "constriction": list(s.constriction),
                "constriction_idx": s.constriction_idx,
                "path": [list(p) for p in s.path],
                "seg1_px": s.seg1_px,
                "seg2_px": s.seg2_px,
                "manual": s.manual,
            }
            for s in seedlings
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_state(path: str | Path) -> tuple[list[Seedling], float, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seedlings = []
    for d in data["seedlings"]:
        seedlings.append(Seedling(
            id=d["id"],
            head=tuple(d["head"]),
            tip=tuple(d["tip"]),
            constriction=tuple(d["constriction"]),
            constriction_idx=d["constriction_idx"],
            path=[tuple(p) for p in d["path"]],
            px_per_cm=data["px_per_cm"],
            seg1_px=d["seg1_px"],
            seg2_px=d["seg2_px"],
            manual=d.get("manual", False),
        ))
    return seedlings, data["px_per_cm"], data["scale"]
