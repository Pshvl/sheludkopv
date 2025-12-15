"""
Модуль image_processing_manual.py

Собственная ручная реализация интерфейса IImageProcessing
без использования библиотеки OpenCV.

Содержит класс ImageProcessingManual, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Собеля)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (Преобразование Хафа)

Модуль предназначен для учебных целей
(лабораторная работа по курсу "Технологии программирования на Python").
"""

import interfaces

import numpy as np


class ImageProcessingManual(interfaces.IImageProcessing):
    """
    Собственная ручная реализация интерфейса IImageProcessing
    без использования библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Sobel).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

        """
        Свертка — это математическая операция, где каждому пикселю выходного изображения
        присваивается сумма произведений пикселей окна входного изображения на
        соответствующие значения ядра.

        Свертка отвечает на вопрос: "Что происходит ВОКРУГ этого пикселя?"
        """

        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        if kernel.ndim != 2:
            raise ValueError("Ядро должно быть двумерным")


        '''
        Вычисляем, сколько пикселей нужно добавить по краям изображения, 
        чтобы ядро могло обработать краевые пиксели.
        '''
        kernel_height, kernel_width = kernel.shape
        pad_h, pad_w = kernel_height // 2, kernel_width // 2

        # Определяем форму выхода
        #обработка различных типов изображений
        '''
            ndim — это атрибут numpy массива, который возвращает количество измерений (размерностей):
            image.ndim == 2 — черно-белое изображение (высота × ширина)
            image.ndim == 3 — цветное изображение (высота × ширина × каналы)
        '''
        #черно белое
        if image.ndim == 2:
            height, width = image.shape
            channels = 1
            image_reshaped = image[:, :, np.newaxis]
            # Добавляем ось каналов для унификации,у RGB 3 канала и будем обрабатывать общим циклом

        #цветное
        elif image.ndim == 3:
            height, width, channels = image.shape
            image_reshaped = image
        else:
            raise ValueError("Unsupported image dimensions")

        # Подготавливаем выходной массив(создание пустого массива)
        output = np.zeros((height, width, channels), dtype=np.float32)


        # Обрабатываем каждый канал
        for ch in range(channels):
            # Берём канал и конвертируем в float32
            # пиксель[red; green; blue]
            #  [:,:, ch] все значения под индексом ch - канал

            channel = image_reshaped[:, :, ch].astype(np.float32)

            # Паддинг с отражением (BORDER_REFLECT_101)
            #используем паддинги которые насчитали сверху
            #к каналу добавляем со всех сторон паддинги (сверху, снизу) (слева, справа) режим отражение краев (нет артефктов и тд)
            padded = np.pad(channel, ((pad_h, pad_h),  (pad_w, pad_w)), mode="reflect")

            # Применяем ядро
            for row in range(height):
                for col in range(width):
                    # Извлекаем окно и вычисляем скалярное произведение
                    #пиксель и его соседи окно скользит по всем
                    window = padded[row: row + kernel_height, col: col + kernel_width]
                   #выходное изобр = окно свертка ядро= суммма (окно * ядро)
                    output[row, col, ch] = np.sum(window * kernel)

        # Убираем лишнюю ось, если было одноканальное изображение
        if image.ndim == 2:
            output = output[:, :, 0]

        return output



    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого
        Args:
            image (np.ndarray): Входное RGB-изображение (H, W, 3).
        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого (H, W),
            dtype=np.uint8.
        """
        #проверка на цветастость и формат
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB with shape (H, W, 3)")
        if image.dtype != np.uint8:
            raise ValueError("Input image must be of type uint8")


        # Точные коэффициенты, как в OpenCV (в float32)
        #извлекаем все каналы по отдельности
        red = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        blue = image[:, :, 2].astype(np.float32)

        # Формула восприятия яркости человеческим глазом. новая чб матрица
        gray = 0.299 * red + 0.587 * green + 0.114 * blue

        # Округление и приведение к uint8
        #диапозон(округ до целых, 0-255, формат) округлили до целых и подогнали до диапозона (другого в изобр не трэба
        return np.clip(np.round(gray), 0, 255).astype(np.uint8)





    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.
        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.
        Используется обратная гамма: I_out = (I_in / 255) ** (1/gamma) * 255.
        Поведение:
          - gamma > 1: осветляет изображение (особенно тёмные области),
          - gamma = 1: изображение не изменяется,
          - 0 < gamma < 1: затемняет изображение (особенно светлые области).
        Args:
            image (np.ndarray): Входное изображение (grayscale или RGB, uint8).
            gamma (float): Значение гамма-коррекции (gamma > 0).
        Returns:
            np.ndarray: Изображение после гамма-коррекции (тот же формат, что и вход).
        Raises:
            ValueError: If gamma is not positive or image is not of type uint8.
        """
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        if image.dtype != np.uint8:
            raise ValueError("Input image must be of type uint8")

        # Обратная гамма
        inv_gamma = 1.0 / gamma

        # Создаём lookup table ( вместо того чтобы постоянно считать формулу
        #мы заранее просчитываем значения для всех 0-255 и делаем таблицу преобразований
        # такому пикселю соответствует такое значение формулы ( не надо считать одинаковые значений н раз)

        lut = np.arange(256, dtype=np.float32) #массив знач пикселей 0-255
        lut = np.power(lut / 255.0, inv_gamma) * 255.0 #применяем формулу ко всем пикселям
        # (пиксель/255.0)^ 1/gamma
        # пиксель/255.0 - нормализуем значение в диапазон [0, 1]
        # ^ 1/gamma - гамма корр
        # * 255 возвращаем в диапозон (0,255)


        #округляем подгоняем под диапозон
        lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)

        # Применяем LUT
        #значения пикселя изображения заменяем на сответсвующий ему по индексу результат вычисления по формуле
        corrected = lut[image]

        return corrected



    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.
        Args:
            image (np.ndarray): Входное изображение.
        Returns:
            np.ndarray: Изображение с выделенными границами.
        Raises:
            ValueError: If the image format is unsupported (not (H, W) or (H, W, 3)).
        """
        if image.ndim == 3 and image.shape[2] == 3:
            gray = self._rgb_to_grayscale(image) #цветное в чб, важна яркость а не цвет
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("Unsupported image format. Expected (H, W) or (H, W, 3).")


        # Ядра Собеля(вклад каждого пикселя в окрестности в конечный результат/дифферен оператор/градиент)
        #по х
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        #по у
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        # Используем _convolution для вычисления градиентов
        gx = self._convolution(gray, sobel_x)
        gy = self._convolution(gray, sobel_y)

        #матрица модулей градиентов (Градиент показывает направление и силу изменения яркости)
        magnitude = np.sqrt(gx**2 + gy**2)

        # Если изображение однородное, нет границ - возвращаем чёрное изображение
        if magnitude.max() == 0:
            return np.zeros_like(magnitude, dtype=np.uint8)

        # Нормализация в [0, 255] и uint8
        #все градиенты/макс градиент (значения 0-1) * 255 (масштабирование 0-255) и в целые числа
        return (magnitude / magnitude.max() * 255).astype(np.uint8)


    #для обнаружения углов
    # Дилатация (расширяем углы, как в OpenCV, усиляем максимумы)
    # Создаём ядро для дилатации 3x3 (как cv2.dilate с None)
    def _dilate_3x3(self, response_map: np.ndarray) -> np.ndarray:
        """Простая дилатация с ядром 3x3 (максимум в окрестности)
        респонс мап - матрица угловатости
        """
        height, width = response_map.shape
        dilated = response_map.copy()
        #по одному пикселю по краям, мод - константы= - бесконечность
        padded = np.pad(response_map, 1, mode="constant", constant_values=- np.inf)

        for row in range(height):
            for col in range(width):
                # Берём максимум в окне 3x3
                window = padded[row: row + 3, col: col + 3]
                #пиксель = макс из его окна = угол расширился на соседние пиксели
                dilated[row, col] = np.max(window)

        return dilated

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении с использованием детектора Харриса.
        Углы выделяются красным цветом на копии исходного изображения,
        как в реализации с использованием cv2.cornerHarris.
        Матрица структуры: M = [I_x**2   I_x*I_y]
                              [I_x*I_y   I_y**2]
        Отклик Харриса: R = det(M) - k * (trace(M))**2
                        det(M) = I_xx*I_yy - I_xy**2
                        trace(M) = I_xx + I_yy
        Args:
            image (np.ndarray): Входное изображение (RGB, uint8).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки, RGB, uint8).

        Raises:
            ValueError: If the input image is not RGB with shape (H, W, 3).
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB with shape (H, W, 3)")

        # Преобразуем в grayscale
        gray = self._rgb_to_grayscale(image)

        # Градиенты Собеля
        # Ядра Собеля(вклад каждого пикселя в окрестности в конечный результат/дифферен оператор/градиент)

        sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                 [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], dtype=np.float32)

        # Используем _convolution для вычисления градиентов(вертикал и горизонт границы, где сильные обе границы это углы)
        gx = self._convolution(gray, sobel_x)
        gy = self._convolution(gray, sobel_y)

        # Компоненты матрицы структуры
        ixx = gx * gx #сила вертикальных
        iyy = gy * gy #горизонтальных
        ixy = gx * gy #заимодействие

        #dst = cv2.cornerHarris(gray, 3, 3, 0.04)
        # Сглаживание (усредняем значения, имитируем blockSize=3 из OpenCV)
        # OpenCV использует усреднение по окну blockSize x blockSize
        box_kernel = np.ones((3, 3), dtype=np.float32)*-1000.0 #ядро усреднения, все 1/9 в суммме 1, чтобы найти реальный угол а не шумы
        print(box_kernel)
        ixx_smooth = self._convolution(ixx, box_kernel)
        iyy_smooth = self._convolution(iyy, box_kernel)
        ixy_smooth = self._convolution(ixy, box_kernel)
        print("ixx",ixx_smooth)
        # Вычисление отклика Харриса
        '''Отклик Харриса: R = det(M) - k * (trace(M))**2
                        det(M) = I_xx*I_yy - I_xy**2
                        trace(M) = I_xx + I_yy
        '''
        harris_k = 0.04
        det_m = ixx_smooth * iyy_smooth - ixy_smooth * ixy_smooth
        print("det", det_m)
        trace_m = ixx_smooth + iyy_smooth
        response = det_m - harris_k * (trace_m**2) #матрица откликов детектора харриса - матрица угловатости

        #матрица с усиленными углами
        response_dilated = self._dilate_3x3(response)
        print("respose", response_dilated)
        # Пороговая обработка (как в OpenCV: > 0.01 * max)
        #result[dst > 0.01 * dst.max()] = [255, 0, 0] #значение > 1% от максимального становится красным
        threshold = 0.01 * response_dilated.max()
        corners_mask = response_dilated > threshold
        #булевая матрица сравнения каждого элемента с одним процентам максимума
        print("corner", corners_mask)
        # Создаём результат с красными точками
        result = image.copy()
        # красными станут только те эллементы, которым в корнер маск соответствовал True
        result[corners_mask] = [255, 0, 0]

        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении
        с использованием преобразования Хафа.

        Метод обнаружения окружностей пока не реализован.

        Args:
            image (np.ndarray): Входное изображение (RGB, uint8).

        Raises:
            NotImplementedError: Метод обнаружения окружностей пока не реализован.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")
