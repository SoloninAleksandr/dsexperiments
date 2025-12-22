#!/usr/bin/env python3
"""
prompt2canvas.py
Запуск:
    python prompt2canvas.py "Текстовый промт" file1.jpg file2.pdf ...
Результат: canvas.png рядом со скриптом.
CLI-скрипт
"""

from pathlib import Path
import argparse, sys

# локальные импорты
import image_conversion as ic
import merge_canvas as mc


def main():
    p = argparse.ArgumentParser(description="Промт + вложения → картинка-холст")
    p.add_argument("prompt", help="Текстовый запрос в кавычках")
    p.add_argument("files", nargs="*", help="Пути к файлам-вложениям")
    p.add_argument("-o", "--out", default="canvas.png", help="Имя выходного файла (PNG/JPG)")
    args = p.parse_args()

    # 1. текст → картинка
    images = [ic.convert_text_to_image(args.prompt)]

    # 2. файлы → картинки
    for f in args.files:
        img = ic.convert_to_image(f)
        if img:
            images.append(img)

    if len(images) == 1:
        print("⚠️  Нет картинок для склейки, сохранился только текст.")
    # 3. склейка
    canvas = mc.stack_images(images)
    mc.save_canvas(canvas, args.out)


if __name__ == "__main__":
    sys.exit(main())
