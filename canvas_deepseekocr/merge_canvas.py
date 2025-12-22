"""
merge_canvas.py
Утилиты для склейки нескольких изображений в один холст.
"""

from __future__ import annotations
from PIL import Image
from typing import List
from pathlib import Path


def stack_images(images: List[Image.Image], bg="white", spacing: int = 10) -> Image.Image:
    """Складывает изображения вертикально с отступом spacing."""
    if not images:
        raise ValueError("Нет изображений для склейки")

    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas_w = max(widths)
    canvas_h = sum(heights) + spacing * (len(images) - 1)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)

    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height + spacing
    return canvas


def save_canvas(canvas: Image.Image, out_path: Path | str):
    """Сохраняет холст в PNG (расширение можно изменить при вызове)."""
    out_path = Path(out_path)
    canvas.save(out_path)
    print(f"🖼  Saved: {out_path.resolve()}")
