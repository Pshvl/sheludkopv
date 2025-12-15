# """
# main.py
#
# Пример лабораторной работы по курсу "Технологии программирования на Python".
#
# Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки OpenCV.
# Реализован консольный интерфейс для применения различных методов обработки к изображению:
# - обнаружение границ (edges)
# - обнаружение углов (corners)
# - обнаружение окружностей (circles)
#
# Запуск:
#     python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]
#
# Аргументы:
#     метод: edges | corners | circles
#     путь_к_изображению: путь к входному изображению
#     -o, --output: путь для сохранения результата (по умолчанию: <имя_входного_файла>_result.png)
#
# Пример:
#     python main.py edges input.jpg
#     python main.py corners input.jpg -o corners_result.png
#
# Автор: [Ваше имя]
# """
#
# import argparse
# import os
#
# import cv2
#
# from implementation import ImageProcessing
#
# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
#     )
#     parser.add_argument(
#         "method",
#         choices=[
#             "edges",
#             "corners",
#             "circles",
#         ],
#         help="Метод обработки: edges, corners, circles",
#     )
#     parser.add_argument(
#         "input",
#         help="Путь к входному изображению",
#     )
#     parser.add_argument(
#         "-o", "--output",
#         help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
#     )
#
#     args = parser.parse_args()
#
#     # Загрузка изображения
#     image = cv2.imread(args.input)
#     if image is None:
#         print(f"Ошибка: не удалось загрузить изображение {args.input}")
#         return
#
#     processor = ImageProcessing()
#
#     # Выбор метода
#     if args.method == "edges":
#         result = processor.edge_detection(image)
#     elif args.method == "corners":
#         result = processor.corner_detection(image)
#     elif args.method == "circles":
#         result = processor.circle_detection(image)
#     else:
#         print("Ошибка: неизвестный метод")
#         return
#
#     # Определение пути для сохранения
#     if args.output:
#         output_path = args.output
#     else:
#         base, ext = os.path.splitext(args.input)
#         output_path = f"{base}_result.png"
#
#     # Сохранение результата
#     cv2.imwrite(output_path, result)
#     print(f"Результат сохранён в {output_path}")
#
#
# if __name__ == "__main__":
#     main()



"""
main.py

Лабораторная работа №1 по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений
с помощью библиотеки OpenCV и собственной ручной реализации.

Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- преобразование в оттенки серого (grayscale)
- гамма-коррекция (gamma)
- свёртка с ядром (conv)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения] [доп. аргументы]

Аргументы:
    метод: edges | corners | grayscale | gamma | conv | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата
    --gamma: коэффициент гамма-коррекции (только для метода gamma, по умолчанию 2.2)
    --kernel: ядро свёртки как строка чисел через запятую
    (только для conv, например: "1,1,1,1,1,1,1,1,1")

Примеры:
    python main.py edges input.jpg
    python main.py corners input.jpg -o result
    python main.py grayscale input.jpg
    python main.py gamma input.jpg --gamma 1.5
    python main.py conv input.jpg --kernel "0,-1,0,-1,4,-1,0,-1,0"
    python main.py circles input.jpg


"""

import argparse
import os
import time

import cv2

from implementation import ImageProcessing

from implementation_manual import ImageProcessingManual

import numpy as np


def main() -> None:
    """
    Основная функция программы.

    Парсит аргументы командной строки, загружает изображение,
    применяет указанный метод обработки с использованием двух реализаций
    (OpenCV и ручной), сохраняет результаты и выводит время выполнения.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Обработка изображения с помощью методов ImageProcessing "
            "(OpenCV и ручная реализация)."
        ),
    )
    parser.add_argument(
        "method",
        choices=["edges", "corners", "grayscale", "gamma", "conv", "circles"],
        help="Метод обработки: edges, corners, grayscale, gamma, conv, circles",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Путь для сохранения результата "
            "(по умолчанию: <input>_result_<метод>_<реализация>)."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Коэффициент гамма-коррекции (только для метода gamma, по умолчанию: 2.2)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help=(
            "Ядро свёртки как строка чисел через запятую "
            "(только для метода conv, например: '0,-1,0,-1,4,-1,0,-1,0')"
        ),
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    # Конвертируем из BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Определение пути для сохранения
    if args.output:
        base_output = args.output
        output_dir = os.path.dirname(base_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        base, _ = os.path.splitext(args.input)
        base_output = f"{base}_result"

    # Обработка изображения с замером времени
    if args.method == "edges":
        start = time.time()
        result_cv = ImageProcessing().edge_detection(image_rgb)
        time_cv = time.time() - start

        start = time.time()
        result_manual = ImageProcessingManual().edge_detection(image_rgb)
        time_manual = time.time() - start

    elif args.method == "corners":
        start = time.time()
        result_cv = ImageProcessing().corner_detection(image_rgb)
        time_cv = time.time() - start

        start = time.time()
        result_manual = ImageProcessingManual().corner_detection(image_rgb)
        time_manual = time.time() - start

    elif args.method == "grayscale":
        start = time.time()
        result_cv = ImageProcessing()._rgb_to_grayscale(image_rgb)
        time_cv = time.time() - start

        start = time.time()
        result_manual = ImageProcessingManual()._rgb_to_grayscale(image_rgb)
        time_manual = time.time() - start

    elif args.method == "gamma":
        start = time.time()
        result_cv = ImageProcessing()._gamma_correction(image_rgb, args.gamma)
        time_cv = time.time() - start

        start = time.time()
        result_manual = ImageProcessingManual()._gamma_correction(image_rgb, args.gamma)
        time_manual = time.time() - start

    elif args.method == "conv":
        if args.kernel is None:
            print("Ошибка: для метода 'conv' требуется аргумент --kernel")
            return

        try:
            kernel_values = list(map(float, args.kernel.split(",")))
            size = len(kernel_values)
            if int(size**0.5) ** 2 != size:
                print(
                    "Ошибка: ядро должно содержать квадратное количество элементов"
                    "(например, 9 для 3x3)",
                )
                return

            kernel = np.array(kernel_values, dtype=np.float32).reshape(int(size**0.5), -1)
        except Exception as e:
            print(f"Ошибка парсинга ядра: {e}")
            return

        start = time.time()
        result_cv = ImageProcessing()._convolution(image_rgb, kernel)
        time_cv = time.time() - start

        start = time.time()
        result_manual = ImageProcessingManual()._convolution(image_rgb, kernel)
        time_manual = time.time() - start

    elif args.method == "circles":
        start = time.time()
        result_cv = ImageProcessing().circle_detection(image_rgb)
        time_cv = time.time() - start

        start = time.time()
        result_manual = ImageProcessingManual().circle_detection(image_rgb)
        time_manual = time.time() - start

    else:
        print("Ошибка: неизвестный метод")
        return

    # Сохранение результатов
    cv2.imwrite(
        f"{base_output}_{args.method}_opencv.png",
        cv2.cvtColor(result_cv, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        f"{base_output}_{args.method}_manual.png",
        cv2.cvtColor(result_manual, cv2.COLOR_RGB2BGR),
    )

    # Вывод результатов
    print(
        f"Результат OpenCV сохранён в "
        f"{base_output}_{args.method}_opencv.png ({time_cv:.3f} сек)",
    )
    print(
        f"Результат ручной реализации сохранён в "
        f"{base_output}_{args.method}_manual.png ({time_manual:.3f} сек)",
    )


if __name__ == "__main__":
    main()
