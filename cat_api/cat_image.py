"""
Модуль cat_image.py

Здесь описаны:
- абстрактный класс CatImage (общий интерфейс для работы с изображениями животных)
- его наследники для цветных и чёрно-белых изображений

CatImage хранит само изображение и метаданные (url, порода),
а также определяет абстрактные методы обработки (detect_edges_custom, detect_edges_library).
Наследники реализуют конкретную логику.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2


class CatImage(ABC):
    """
    Абстрактный класс для работы с изображениями животных.

    От этого класса наследуются конкретные реализации:
    - ColorCatImage  — цветные изображения
    - GrayscaleCatImage — ч/б изображения

    Абстрактный значит, что напрямую CatImage создавать нельзя,
    нужно использовать один из наследников.
    """

    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        """
        Конструктор (метод __init__).

        Вызывается при создании объекта:
            obj = ColorCatImage(image_data, url, breed)

        Здесь мы сохраняем:
        - само изображение (numpy-массив)
        - ссылку на изображение
        - текстовое название породы
        """
        self._image_data = image_data
        self._image_url = image_url
        self._breed = breed

    @property
    def image_data(self) -> np.ndarray:
        """Свойство (property) для чтения изображения как атрибута: obj.image_data."""
        return self._image_data

    @property
    def image_url(self) -> str:
        """Свойство для чтения URL изображения: obj.image_url."""
        return self._image_url

    @property
    def breed(self) -> str:
        """Свойство для чтения породы: obj.breed."""
        return self._breed

    # --- Общие методы обработки (скопированы из image_processing_manual.py) ---
    # Теперь они являются частью CatImage и доступны наследникам

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # (Вставь сюда код из image_processing_manual.py)
        # Пример:
        if kernel.ndim != 2:
            raise ValueError("Ядро должно быть двумерным")

        kernel_height, kernel_width = kernel.shape
        pad_h, pad_w = kernel_height // 2, kernel_width // 2

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
            image_reshaped = image[:, :, np.newaxis]
        elif image.ndim == 3:
            height, width, channels = image.shape
            image_reshaped = image
        else:
            raise ValueError("Unsupported image dimensions")

        output = np.zeros((height, width, channels), dtype=np.float32)

        for ch in range(channels):
            channel = image_reshaped[:, :, ch].astype(np.float32)
            padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

            for row in range(height):
                for col in range(width):
                    window = padded[row: row + kernel_height, col: col + kernel_width]
                    output[row, col, ch] = np.sum(window * kernel)

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
      
    def _edge_detection_impl(self, image: np.ndarray) -> np.ndarray:
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



            # --- Абстрактные методы, которые должны реализовать наследники ---
    @abstractmethod
    def detect_edges_custom(self) -> np.ndarray:
        """Абстрактный метод для пользовательского обнаружения границ."""
        # Этот метод будет использовать _edge_detection_impl
        pass

    @abstractmethod
    def detect_edges_library(self) -> np.ndarray:
        """Абстрактный метод для библиотечного обнаружения границ (Canny)."""
        pass

    # # --- Перегрузка операторов ---
    # def __add__(self, other: 'CatImage') -> 'CatImage':
    #     img1 = self._image_data
    #     img2 = other._image_data

    #     max_h = max(img1.shape[0], img2.shape[0])
    #     max_w = max(img1.shape[1], img2.shape[1])

    #     # Вычисляем паддинги
    #     pad_h1 = max_h - img1.shape[0]
    #     pad_w1 = max_w - img1.shape[1]
    #     pad_h2 = max_h - img2.shape[0]
    #     pad_w2 = max_w - img2.shape[1]

    #     # Паддинг для первого изображения
    #     img1_padded = np.pad(img1, ((0, pad_h1), (0, pad_w1)), mode='reflect')
    #     # Паддинг для второго изображения
    #     img2_padded = np.pad(img2, ((0, pad_h2), (0, pad_w2)), mode='reflect')

    #     # Сложение
    #     #поэлементное сложение двух массивов
    #     combined = np.clip(img1_padded.astype(int) + img2_padded.astype(int), 0, 255).astype(np.uint8)
    #     return self.__class__(combined, f"{self.breed}_plus_{other.breed}", "combined")

    # def __sub__(self, other: 'CatImage') -> 'CatImage':
    #     img1 = self._image_data
    #     img2 = other._image_data

    #     max_h = max(img1.shape[0], img2.shape[0])
    #     max_w = max(img1.shape[1], img2.shape[1])

    #     # Вычисляем паддинги
    #     pad_h1 = max_h - img1.shape[0]
    #     pad_w1 = max_w - img1.shape[1]
    #     pad_h2 = max_h - img2.shape[0]
    #     pad_w2 = max_w - img2.shape[1]

    #     # Паддинг для первого изображения
    #     img1_padded = np.pad(img1, ((0, pad_h1), (0, pad_w1)), mode='reflect')
    #     # Паддинг для второго изображения
    #     img2_padded = np.pad(img2, ((0, pad_h2), (0, pad_w2)), mode='reflect')

    #     # Вычитание
    #     combined = np.clip(img1_padded.astype(int) - img2_padded.astype(int), 0, 255).astype(np.uint8)
    #     return self.__class__(combined, f"{self.breed}_minus_{other.breed}", "subtracted")


    def __add__(self, other: 'CatImage') -> 'CatImage':
        img1 = self._image_data
        img2 = other._image_data

        # Проверяем
        if img1.ndim != img2.ndim:
            if img1.ndim == 3 and img2.ndim == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
                print("  → Правый операнд приведён к цветному")
            elif img1.ndim == 2 and img2.ndim == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                print("  → Правый операнд приведён к ч/б")

        max_h = max(img1.shape[0], img2.shape[0])
        max_w = max(img1.shape[1], img2.shape[1])

        # Вычисляем паддинги
        pad_h1 = max_h - img1.shape[0]
        pad_w1 = max_w - img1.shape[1]
        pad_h2 = max_h - img2.shape[0]
        pad_w2 = max_w - img2.shape[1]

        # Для цветных (3D) и ч/б (2D) разный padding
        if img1.ndim == 3:
            # Цветное: (h, w, 3)
            img1_padded = np.pad(img1, ((0, pad_h1), (0, pad_w1), (0, 0)), mode='reflect')
            img2_padded = np.pad(img2, ((0, pad_h2), (0, pad_w2), (0, 0)), mode='reflect')
        else:
            # Ч/б: (h, w)
            img1_padded = np.pad(img1, ((0, pad_h1), (0, pad_w1)), mode='reflect')
            img2_padded = np.pad(img2, ((0, pad_h2), (0, pad_w2)), mode='reflect')

        # Сложение
        combined = np.clip(img1_padded.astype(int) + img2_padded.astype(int), 0, 255).astype(np.uint8)
        
        # Возвращаем того же типа что и первый операнд
        return self.__class__(combined, 
                          f"combined_from_{self.image_url}_and_{other.image_url}", 
                          f"({self.breed})_plus_({other.breed})")

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        img1 = self._image_data
        img2 = other._image_data

        # Проверяем
        if img1.ndim != img2.ndim:
            if img1.ndim == 3 and img2.ndim == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
                print("  → Правый операнд приведён к цветному")
            elif img1.ndim == 2 and img2.ndim == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                print("  → Правый операнд приведён к ч/б")

        max_h = max(img1.shape[0], img2.shape[0])
        max_w = max(img1.shape[1], img2.shape[1])

        # Вычисляем паддинги
        pad_h1 = max_h - img1.shape[0]
        pad_w1 = max_w - img1.shape[1]
        pad_h2 = max_h - img2.shape[0]
        pad_w2 = max_w - img2.shape[1]

        # Для цветных (3D) и ч/б (2D) разный padding
        if img1.ndim == 3:
            # Цветное: (h, w, 3)
            img1_padded = np.pad(img1, ((0, pad_h1), (0, pad_w1), (0, 0)), mode='reflect')
            img2_padded = np.pad(img2, ((0, pad_h2), (0, pad_w2), (0, 0)), mode='reflect')
        else:
            # Ч/б: (h, w)
            img1_padded = np.pad(img1, ((0, pad_h1), (0, pad_w1)), mode='reflect')
            img2_padded = np.pad(img2, ((0, pad_h2), (0, pad_w2)), mode='reflect')

        # Вычитание
        combined = np.clip(img1_padded.astype(int) - img2_padded.astype(int), 0, 255).astype(np.uint8)
        
        # Возвращаем того же типа что и первый операнд
        return self.__class__(combined, 
                          f"subtracted_from_{self.image_url}_and_{other.image_url}", 
                          f"({self.breed})_minus_({other.breed})")
    def __str__(self) -> str:
        return f"CatImage(breed={self._breed}, url={self._image_url})"


class ColorCatImage(CatImage):
    """Класс для цветных изображений."""

    def detect_edges_custom(self) -> np.ndarray:
        """Выделение контуров пользовательским методом (Sobel)."""
        # Используем внутреннюю реализацию, передав ей цветное изображение
        # Оно будет автоматически преобразовано в серое внутри _edge_detection_impl
        return self._edge_detection_impl(self._image_data)

    def detect_edges_library(self) -> np.ndarray:
        """Выделение контуров библиотечным методом (Canny)."""
        # Преобразуем в градации серого с помощью cv2
        gray_image = cv2.cvtColor(self._image_data, cv2.COLOR_RGB2GRAY) # Предполагаем RGB
        edges = cv2.Canny(gray_image, 100, 200)
        return edges


class GrayscaleCatImage(CatImage):
    """Класс для ч/б изображений."""

    def detect_edges_custom(self) -> np.ndarray:
        """Выделение контуров пользовательским методом (Sobel)."""
        # Используем внутреннюю реализацию, передав ей ч/б изображение
        return self._edge_detection_impl(self._image_data)

    def detect_edges_library(self) -> np.ndarray:
        """Выделение контуров библиотечным методом (Canny)."""
        # self._image_data уже ч/б
        edges = cv2.Canny(self._image_data, 100, 200)
        return edges