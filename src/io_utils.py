from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False


def load_bgr(path: str | Path) -> np.ndarray:
    """Carrega uma imagem em BGR. Aceita .heic/.heif via pillow-heif."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        if not _HEIC_OK:
            raise RuntimeError(
                "Suporte a HEIC indisponivel. Instale pillow e pillow-heif."
            )
        img = Image.open(path).convert("RGB")
        rgb = np.array(img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {path}")
    return data


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
