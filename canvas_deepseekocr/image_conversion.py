"""
image_conversion.py
Преобразования:
1. текст  → изображение (Pillow)
2. файл   → изображение  (фото / первая страница PDF / первое изображение из ZIP)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List

from PIL import Image, ImageDraw, ImageFont
import io, zipfile, mimetypes

# ---------- 1. ТЕКСТ → КАРТИНКА ---------- #
def convert_text_to_image(text: str, width: int = 800, bg="white", fg="black") -> Image.Image:
    """Рендерит многострочный text в PNG-картинку с автопереносом."""
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: List[str] = []
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        w_len, _ = draw.textsize(test, font)
        if w_len <= width - 40:  # отступы
            line = test
        else:
            lines.append(line)
            line = w
    lines.append(line)

    line_height = draw.textsize("A", font)[1] + 4
    height = line_height * len(lines) + 40
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    y = 20
    for l in lines:
        draw.text((20, y), l, fill=fg, font=font)
        y += line_height
    return img


# ---------- 2. ФАЙЛ → КАРТИНКА ---------- #
def _pdf_first_page_to_image(pdf_path: Path) -> Optional[Image.Image]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❗ Установи пакет 'pymupdf' для конвертации PDF.")
        return None

    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap()
            data = pix.tobytes("png")
            return Image.open(io.BytesIO(data))
    except Exception as e:
        print(f"PDF-ошибка: {e}")
        return None


def _first_image_from_zip(zip_path: Path) -> Optional[Image.Image]:
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    with z.open(name) as file:
                        return Image.open(io.BytesIO(file.read()))
    except Exception as e:
        print(f"ZIP-ошибка: {e}")
    return None


def convert_to_image(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Поддержка:
      • изображения (.png .jpg .jpeg .gif …)
      • PDF (первая страница)
      • ZIP-архив (первое изображение внутри)
    Возвращает PIL.Image или None.
    """
    path = Path(path)
    if not path.exists():
        print(f"Файл не найден: {path}")
        return None

    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".gif"):
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Не могу открыть картинку: {e}")
            return None

    if suffix == ".pdf":
        return _pdf_first_page_to_image(path)

    if suffix == ".zip":
        return _first_image_from_zip(path)

    # Попытка по MIME
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Не могу открыть (MIME-image): {e}")

    print(f"Тип файла не поддержан: {path.name}")
    return None
