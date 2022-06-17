

import numpy as np
from functions import shapefile



def scale_shapefile_elements(lines_sample):
    """
    Переводит координаты трещин из shape файла к реальному масштабу (в миллиметры)
    """
    return np.array(lines_sample) * 0.25 / 100



def truncate_image(lines_sample,
                   left, right, bottom, top):
    """
    Вырезает прямоугольную область шлифа по заданным границам

    :param lines_sample: выборка координат трещин
    :param left: левая граница в миллиметрах
    :param right: правая граница в миллиметрах
    :param bottom: нижняя граница в миллиметрах
    :param top: верхняя граница в миллиметрах
    :return: выборка координат трещин для обрезанного образца
    """
    new_lines_sample = []

    for x1, y1, x2, y2 in lines_sample:

        flag1 = (x1 > left) and (x2 > left)
        flag2 = (x1 < right) and (x2 < right)
        flag3 = (y1 > bottom) and (y2 > bottom)
        flag4 = (y1 < top) and (y2 < top)

        if np.all([flag1, flag2, flag3, flag4]):
            new_lines_sample.append((x1, y1, x2, y2))

    return new_lines_sample




def read_shape_file(file_path,
                    left, right, bottom, top):
    """
    Читает shape файл
    Возвращает список всех трещин в формате (x_start, y_start, x_end, y_end)
    Обрезает края шлифа на 5% со всех сторон
    """
    # чтение
    sf = shapefile.Reader(file_path)
    shapes = sf.shapes()
    n = len(shapes)

    # извлечение координат
    X = np.zeros(n, dtype='object')
    Y = np.zeros(n, dtype='object')

    for i in range(n):
        points = shapes[i].points
        X[i] = []
        Y[i] = []
        for point in points:
            X[i].append(point[0])
            Y[i].append(point[1])

    points = []
    for i in range(len(X)):
        points.append([])
        for j in range(len(X[i])):
            points[-1].append((X[i][j], Y[i][j]))

    # запись координат в удобном формате
    lines_sample = []
    for k in range(len(points)):
        for i in range(len(points[k])):
            if i < len(points[k]) - 1:
                x1, y1 = points[k][i]
                x2, y2 = points[k][i + 1]
                lines_sample.append((x1, y1, x2, y2))

    # перевод из пикселей в миллиметры
    lines_sample = scale_shapefile_elements(lines_sample)
    # обрезка краев
    lines_sample = truncate_image(lines_sample,
                                  left, right, bottom, top)
    lines_sample = np.array(lines_sample)

    return lines_sample

