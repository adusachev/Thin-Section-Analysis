import numpy as np
from functions import shapefile
from numba import njit



def get_lines(path):
    """
    :param path: путь до .shp файла
    :return: выборка линеаментов
                    (каждый линеамент задан как список его вершин,
                     e.g. [(x1, y1), (x2, y2), (x3, y3)])
    """
    sf = shapefile.Reader(path)
    shapes = sf.shapes()
    n = len(shapes)

    lines = []
    for i in range(n):
        lines.append(shapes[i].points)

    return lines



def pixels_to_mm(value):
    """
    Переводит значение в пикселях в миллиметры
    :param value: значение (тип int, float, np.array)
    :return: value в миллиметрах
    """
    return (value * 0.25) / 100


def get_segments(line):
    """
    Возвращает список сегментов данного линеамента

    :param line: список вешрин линеамента  // e.g. [(x1, y1), (x2, y2), (x3, y3)]
    :return: cписок сегментов данного линеамента, где каждый сегмент
             задается координатами начала и конца отрезка,
             // e.g. np.array([[x1, y1, x2, y2],
                               [x2, y2, x3, y3]])
    """
    n = len(line)
    line_segments = [[line[0][0], line[0][1]]]

    for i in range(1, n- 1):
        line_segments[-1].append(line[i][0])
        line_segments[-1].append(line[i][1])
        line_segments.append([])
        line_segments[-1].append(line[i][0])
        line_segments[-1].append(line[i][1])

    line_segments[-1].append(line[-1][0])
    line_segments[-1].append(line[-1][1])

    return np.array(line_segments)



# def get_lines_segments_sample(path_to_shp):
#     """
#     Возвращает список линеаментов в shape-файле, где каждый линеамент задан списком сегментов,
#     а каждый сегмент задан координатами x, y начала и конца
#
#     :param path_to_shp: путь до shape-файла
#     :return: lines_segments - список, каждый элемент которого это список сегментов одного линеамента
#                                // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]]) - один эл-т lines_segments
#     """
#     # чтение
#     sf = shapefile.Reader(path_to_shp)
#     shapes = sf.shapes()
#     n = len(shapes)
#
#     # вершины линеаментов
#     lines = []
#     for i in range(n):
#         lines.append(shapes[i].points)
#
#     # сегменты линеаментов
#     lines_segments = []
#     for line in lines:
#         line_seg = get_segments(line)
#         lines_segments.append(line_seg)
#     for i in range(len(lines_segments)):
#         lines_segments[i] = np.array(lines_segments[i])
#
#     return lines_segments


def get_lines_segments_sample(lines):
    """
    Возвращает список линеаментов в shape-файле, где каждый линеамент задан списком сегментов,
    а каждый сегмент задан координатами x, y начала и конца

    :param lines: cписок сегментов всех линеаментов, где каждый сегмент
                  задается координатами начала и конца отрезка,
                    // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]])
    :return: lines_segments - список, каждый элемент которого это список сегментов одного линеамента
                               // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]]) - один эл-т lines_segments
    """

    # сегменты линеаментов
    lines_segments = []
    for line in lines:
        line_seg = get_segments(line)
        lines_segments.append(line_seg)
    for i in range(len(lines_segments)):
        lines_segments[i] = np.array(lines_segments[i])

    return lines_segments





@njit
def dist(point1, point2):
    """
    Возвращает расстояние между двумя точками на плоскости
    :param point1:
    :param point2:
    :return: d - евклидово расстояние между point1 и point2
    """
    x1, y1 = point1
    x2, y2 = point2

    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return d

@njit
def trace_length(line_segments):
    """
    Возвращает длину линеамента
    :param line_segments: cписок сегментов данного линеамента  // e.g. np.array([[x1, y1, x2, y2],
                                                                                 [x2, y2, x3, y3]])
    :return: length - сумма длин всех сегментов линеамента
    """
    length = 0

    for seg in line_segments:
        seg_length = dist(seg[:2], seg[2:])
        length += seg_length

    return length


@njit
def trace_length_sample(lines_segments):
    """
    Возврвщает выборку длин линеаментов по всему шлифу
    :param lines_segments: список линеаментов, где каждый линеамент задается как cписок сегментов данного линеамента
    :return: length_sample - массив длин линеаментов
    """
    n = len(lines_segments)
    length_sample = np.zeros(n)

    for i in range(n):
        length_sample[i] = trace_length(lines_segments[i])

    return length_sample

