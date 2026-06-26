"""Converte uma imagem HEIC para PNG (e gera uma versao reduzida para inspecao)."""
import sys
from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    img = Image.open(src)
    img = img.convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "PNG")
    print(f"Salvo {dst} tamanho={img.size}")

    # versao reduzida para inspecao rapida
    preview = img.copy()
    preview.thumbnail((1400, 1400))
    prev_path = dst.with_name(dst.stem + "_preview.png")
    preview.save(prev_path, "PNG")
    print(f"Preview {prev_path} tamanho={preview.size}")


if __name__ == "__main__":
    main()
