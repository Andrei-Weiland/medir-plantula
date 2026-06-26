"""Medicao de plantulas de alface (Computacao Grafica / OpenCV).

Mede, para cada plantula, dois segmentos seguindo o caminho real (curvo):
  - Segmento 1: topo da estrutura branca  ->  ponto de estrangulamento
  - Segmento 2: ponto de estrangulamento  ->  extremidade da raiz
Converte para cm via regua (automatico) ou cliques, e exporta tabela + imagem
anotada.

Uso tipico (modo hibrido com correcao manual):
    python main.py --imagem data/entrada/IMG_3265.png

Modo automatico sem janela (headless):
    python main.py --imagem data/entrada/IMG_3265.png --sem-gui
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.io_utils import load_bgr, save_image
from src import pipeline as pl
from src import visualize as vz
from src import export as ex
from src.calibration import from_ruler


def parse_args():
    p = argparse.ArgumentParser(description="Medicao de plantulas de alface")
    p.add_argument("--imagem", required=True, help="Imagem de entrada (HEIC/PNG/JPG)")
    p.add_argument("--saida", default="data/saida", help="Pasta de saida")
    p.add_argument("--max-dim", type=int, default=1400,
                   help="Maior dimensao da resolucao de trabalho")
    p.add_argument("--px-por-cm", type=float, default=None,
                   help="Escala em px/cm (na imagem original). Ignora a regua.")
    p.add_argument("--calibracao", choices=["regua", "cliques"], default="regua",
                   help="Metodo de calibracao inicial")
    p.add_argument("--sem-gui", action="store_true",
                   help="Roda so o automatico e salva (sem janela interativa)")
    return p.parse_args()


def resolve_scale(args, work, bgr_orig) -> float:
    """Determina px/cm na resolucao de TRABALHO."""
    if args.px_por_cm:
        ppc_full = args.px_por_cm
        print(f"Escala informada: {ppc_full:.1f} px/cm (original)")
        return ppc_full * work.scale

    if args.calibracao == "regua":
        cal = from_ruler(bgr_orig)
        if cal is not None:
            print(f"Regua detectada: {cal.px_per_cm:.1f} px/cm "
                  f"(confianca {cal.confidence}, {cal.detail})")
            return cal.px_per_cm * work.scale
        print("Nao foi possivel ler a regua automaticamente.")

    # fallback: calibrar por cliques (no GUI) ou avisar (headless)
    return 0.0


def main():
    args = parse_args()
    img_path = Path(args.imagem)
    stem = img_path.stem

    print(f"Carregando {img_path} ...")
    bgr_orig = load_bgr(img_path)
    work = pl.prepare(bgr_orig, max_dim=args.max_dim)
    print(f"ROI do papel: {work.roi}  | escala de trabalho: {work.scale:.3f}")

    px_per_cm = resolve_scale(args, work, bgr_orig)

    if args.sem_gui:
        if px_per_cm <= 0:
            print("AVISO: sem calibracao valida; medidas sairao apenas em px "
                  "(use --px-por-cm ou --calibracao regua).")
            px_per_cm = 1e-9  # evita divisao por zero; cm ~ px
        seedlings = pl.auto_detect(work, px_per_cm)
        print(f"Detectadas {len(seedlings)} plantulas (automatico).")
        seedlings.sort(key=lambda s: (s.head[1], s.head[0]))
        for i, s in enumerate(seedlings):
            s.id = i
        out_dir = Path(args.saida)
        annotated = vz.annotate(work.bgr, seedlings, work.roi)
        save_image(out_dir / f"{stem}_anotado.png", annotated)
        ex.save_csv(seedlings, out_dir / f"{stem}.csv")
        ex.save_state(seedlings, px_per_cm, work.scale,
                      out_dir / f"{stem}_estado.json")
        print(f"Resultados salvos em {out_dir}")
        for s in seedlings:
            print(f"  #{s.id+1}: seg1={s.seg1_cm:.2f}  seg2={s.seg2_cm:.2f}  "
                  f"total={s.total_cm:.2f} cm")
        return

    # modo interativo (hibrido)
    from src.interactive import Editor
    editor = Editor(work, px_per_cm if px_per_cm > 0 else 1e-9,
                    args.saida, stem=stem)
    print("Rodando deteccao automatica inicial ...")
    editor.run_auto()
    if px_per_cm <= 0:
        print("Calibre a escala: tecle 'k' e clique em duas marcas de cm.")
    print("Abrindo janela interativa. Tecle 'h' para ajuda.")
    editor.loop()


if __name__ == "__main__":
    main()
