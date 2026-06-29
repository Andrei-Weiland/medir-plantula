from __future__ import annotations

import argparse
from pathlib import Path

from src.io_utils import load_bgr
from src import pipeline as pl
from src.calibration import from_ruler

SAIDA = "data/saida"
MAX_DIM = 1400

def parse_args():
    p = argparse.ArgumentParser(description="Medicao de plantulas de alface")
    p.add_argument("--imagem", required=True, help="Imagem de entrada (HEIC/PNG/JPG)")
    return p.parse_args()


def resolve_scale(work, bgr_orig) -> float:
    """Determina px/cm na resolucao de trabalho (regua automatica)."""
    cal = from_ruler(bgr_orig)
    if cal is not None:
        print(f"Regua detectada: {cal.px_per_cm:.1f} px/cm "
              f"(confianca {cal.confidence}, {cal.detail})")
        return cal.px_per_cm * work.scale
    print("Nao foi possivel ler a regua automaticamente.")
    return 0.0


def main():
    args = parse_args()
    img_path = Path(args.imagem)
    stem = img_path.stem

    print(f"Carregando {img_path} ...")
    bgr_orig = load_bgr(img_path)
    work = pl.prepare(bgr_orig, max_dim=MAX_DIM)
    print(f"ROI do papel: {work.roi}  | escala de trabalho: {work.scale:.3f}")

    px_per_cm = resolve_scale(work, bgr_orig)

    from src.interactive import Editor
    editor = Editor(work, px_per_cm if px_per_cm > 0 else 1e-9,
                    SAIDA, stem=stem)
    print("Rodando deteccao automatica inicial ...")
    editor.run_auto()
    if px_per_cm <= 0:
        print("Calibre a escala: tecle 'k' e clique em duas marcas de cm.")
    print("Abrindo janela interativa. Tecle 'h' para ajuda.")
    editor.loop()


if __name__ == "__main__":
    main()
