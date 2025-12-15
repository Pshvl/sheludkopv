

#
# # 1.
# cat1_image = cv2.imread("cat1.jpeg")
# cat2_image = cv2.imread("cat2.jpg")
# cat3_image=cv2.imread("cat3.jpg")
#
# # 2. Конвертируем BGR → RGB
# cat1_image_rgb = cv2.cvtColor(cat1_image, cv2.COLOR_BGR2RGB)
# cat2_image_rgb = cv2.cvtColor(cat2_image, cv2.COLOR_BGR2RGB)
# cat3_image_rgb = cv2.cvtColor(cat3_image, cv2.COLOR_BGR2RGB)
#
# # 3. Создаем цветные объекты (ColorCatImage)
# cat1 = ColorCatImage(cat1_image_rgb, "cat1.jpg", "Первая кошка")
# cat2 = ColorCatImage(cat2_image_rgb, "cat2.jpg", "Вторая кошка")
# cat3 = ColorCatImage(cat3_image_rgb, "cat3.jpg", "Пятая кошка")
#
# cat3_edges= cat3.detect_edges_custom()
# result_plus = cat1 + cat2
#
# print(f"Порода результата сложения: {result_plus.breed}")
# # Сохраняем результат (конвертируем обратно RGB → BGR для OpenCV)
# cv2.imwrite("plus.jpeg", cv2.cvtColor(result_plus._image_data, cv2.COLOR_RGB2BGR))
#
# result_plus3 = cat2 + cat3
# print(f"Порода результата сложения: {result_plus3.breed}")
# # Сохраняем результат (конвертируем обратно RGB → BGR для OpenCV)
# cv2.imwrite("plus3.jpeg", cv2.cvtColor(result_plus3._image_data, cv2.COLOR_RGB2BGR))
#
# result_minus = cat1 - cat2
# print(f"Порода результата вычитания: {result_minus.breed}")
# cv2.imwrite("minus.jpeg", cv2.cvtColor(result_minus._image_data, cv2.COLOR_RGB2BGR))
#
# # 5. Для черно-белых изображений нужно конвертировать в grayscale
#
# cat1_gray = cv2.cvtColor(cat1_image, cv2.COLOR_BGR2GRAY)  # BGR напрямую в grayscale
# cat2_gray = cv2.cvtColor(cat2_image, cv2.COLOR_BGR2GRAY)
#
#
#
# # 6. Создаем черно-белые объекты (GrayscaleCatImage)
# cat3 = GrayscaleCatImage(cat1_gray, "cat3.jpg", "Третья кошка")
# cat4 = GrayscaleCatImage(cat2_gray, "cat2.jpg", "Четвертая кошка")
# #result_plus3 = cat3 + cat3_edges
# # 7.
# result_plus2 = cat3 + cat4
# print(f"Порода результата сложения grayscale: {result_plus2.breed}")
# # Для grayscale изображения не нужно конвертировать цвет
# cv2.imwrite("plus2.jpeg", result_plus2._image_data)
#
# result_minus2 = cat3 - cat4
# print(f"Порода результата вычитания grayscale: {result_minus2.breed}")
# cv2.imwrite("minus2.jpeg", result_minus2._image_data)
# cv2.imwrite("plusedge.jpeg", result_plus3._image_data)

# cat3_edge=cv2.imread("cat3.jpg")#загружаем
# cat3_edge = cv2.cvtColor(cat3_edge, cv2.COLOR_BGR2GRAY)
# cat3_edge_original = GrayscaleCatImage(cat3_edge, "cat3.jpg", "Первая кошка")
# cat3_edge= cat3_edge_original.detect_edges_custom()
# cv2.imwrite("cat3_edge.jpeg", cat3_edge._image_data)

import cv2
from cat_image import ColorCatImage, GrayscaleCatImage
#загрузили
cat_image = cv2.imread("cat3.jpg")
#объект
color_cat = ColorCatImage(cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB),"cat3.jpg", "Кошка")
#границы
edges_array = color_cat.detect_edges_custom()
#границы в объект
edges_gray = GrayscaleCatImage(edges_array, "edges.jpg", "Границы")

print("=== Сложение цветной кошки с ч/б границами = цветная кошка ===")
result1 = color_cat + edges_gray
print(f"Результат: {type(result1).__name__} - {result1.breed}")
cv2.imwrite("cat3_1_with_edges.jpg", cv2.cvtColor(result1._image_data, cv2.COLOR_RGB2BGR))

print("=== Сложение ч/б границ с цветной кошкой = ч/б кошка ===")
result2 = edges_gray+color_cat
print(f"Результат: {type(result2).__name__} - {result2.breed}")
cv2.imwrite("cat3_2_with_edges.jpg", cv2.cvtColor(result2._image_data, cv2.COLOR_RGB2BGR))