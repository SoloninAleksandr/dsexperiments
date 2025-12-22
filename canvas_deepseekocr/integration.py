"""
integration.py
пример, как можно объединить функции конвертации и склейки
в едином вызове create_canvas(prompt, files).
"""

from pathlib import Path
from typing import List
from .image_conversion import convert_text_to_image, convert_to_image
from .merge_canvas import stack_images, save_canvas


def create_canvas(prompt: str, files: List[Path | str], out: Path | str) -> Path:
    imgs = [convert_text_to_image(prompt)]
    for f in files:
        img = convert_to_image(Path(f))
        if img:
            imgs.append(img)
    canvas = stack_images(imgs)
    save_canvas(canvas, out)
    return Path(out)
