"""Converte uma imagem HEIC para PNG"""
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

if __name__ == "__main__":
    main()
