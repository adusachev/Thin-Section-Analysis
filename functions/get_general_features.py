
import numpy as np


def length(line):
    """
    Возвращает длину сегмента
    :param line: один сегмент в формате (x_start, y_start, x_end, y_end)
    """
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def azimuth(line):
    """
    Вычисляет угол между сегментом и осью х
    :param line: один сегмент в формате (x_start, y_start, x_end, y_end)
    :return: угол наклона сегмента в диапазоне (0, pi]
    """
    x1, y1, x2, y2 = line
    cathet_1 = x2 - x1
    cathet_2 = y2 - y1

    if cathet_1 != 0:
        angle = np.arctan(cathet_2 / cathet_1)
        # чтобы было в диапазоне (0, pi)
        if angle < 0:
            angle = np.pi - np.abs(angle)
    else:
        angle = np.pi / 2

    # объединяем 0 и pi
    if angle == 0:
        angle = np.pi

    return angle



def get_length_sample(lines_sample):
    """Вычисляет длины всех сегментов выборки"""
    n = len(lines_sample)
    length_sample = np.zeros(n)

    for i in range(n):
        line = lines_sample[i]
        length_sample[i] = length(line)

    return length_sample



def get_azimuth_sample_rad(lines_sample):
    """Вычисляет углы наклона всех сегментов выборки (в радианах)"""
    n = len(lines_sample)
    azimuth_sample = np.zeros(n)

    for i in range(n):
        line = lines_sample[i]
        azimuth_sample[i] = azimuth(line)

    return azimuth_sample


def lineament_azimuth(line_segments, rad=False):
    """
    Вычисляет азимут одного линеамента как угол наклона прямой,
    соединяющей его начало и конец (спрямленного линеамента)

    :param line_segments: список сегментов линеамента  // e.g. np.array([[x1, y1, x2, y2],
                                                                        [x2, y2, x3, y3]])
    :param rad: возвращает азимуты в радианах, если True
    :return: угол наклона линеамента
    """
    x_start, y_start = line_segments[0][:2]
    x_end, y_end = line_segments[-1][-2:]

    az = azimuth((x_start, y_start, x_end, y_end))
    if rad:
        return az
    return az * (180 / np.pi)



def get_lineament_azimuth_sample(lineaments_sample, rad=False):
    """
    Возвращает выборку азимутов линеаментов для одного шлифа

    :param lineaments_sample: список линеаментов, где каждый линеамент задается как cписок сегментов данного линеамента
    :param rad: возвращает азимуты в радианах, если True
    """
    n = len(lineaments_sample)
    lineament_azimuth_sample = np.zeros(n)

    for i, line_segments in enumerate(lineaments_sample):
        lineament_azimuth_sample[i] = lineament_azimuth(line_segments, rad)

    return lineament_azimuth_sample




def get_center(lines_sample):
    """
    Вычисляет координаты центров каждого отрезка
    Возвращает выборку c элементами (x_center, y_center)
    """
    n = len(lines_sample)
    center_sample = np.zeros((n, 2))
    for i in range(n):
        x1, y1, x2, y2 = lines_sample[i]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2

        center_sample[i, 0] = x_c
        center_sample[i, 1] = y_c

    return center_sample



def choose_window_size(length_sample):
    """
    Выбирает оптимальный размер окна сканирования для вычисления плотности трещин
    Берет 99% от отсортированной выборки длин, и выбирает размер окна равный
     максимальному элементу этой выборки и умножает его на 1.1
    :param length_sample: выборка длин сегментов трещин
    :return: размер окна
    """
    sorted_sample = np.sort(length_sample)
    n = len(sorted_sample)
    m = int(0.99 * n)
    window_size = 1.1 * sorted_sample[m-1]
    return window_size



def compute_density(lines_sample, k=None):
    """
    Вычисляет распределение плотности сети трещин
    Едет по изображению ячейкой k x k
    Плотность в каждой ячейке = число сегментов в ячейке / (k^2)

    :param lines_sample: выборка сегментов в формате (x_start, y_start, x_end, y_end)
    :param k: размер движущейся ячейки
    :return: двумерный массив плотности трещин

    P.S. окно начинает движение с верхнего левого угла и едет вправо
        при постоянном y-ке до предела, потом опускается и тд

    P.P.S. возвращает x_iter, y_iter для того чтобы позже решейпить массив density_sample
            и отрисовывать карту плотности
    """
    # выбор ширины окна (если не задано вручную)
    if k is None:
        length_sample = get_length_sample(lines_sample)
        k = choose_window_size(length_sample)

    # локализация изображения
    # lines_sample = np.array(lines_sample)
    x_sample = np.hstack((lines_sample[:, 0], lines_sample[:, 2]))  # выборка ВСЕХ х-ов
    xmax = np.max(x_sample)
    xmin = np.min(x_sample)
    y_sample = np.hstack((lines_sample[:, 1], lines_sample[:, 3]))  # выборка ВСЕХ y-ов
    ymax = np.max(y_sample)
    ymin = np.min(y_sample)

    # границы сканируемой области:
    x_start, x_end = xmin, xmax
    y_start, y_end = ymin, ymax
    # текущие координаты окна (координаты левого нижнего угла)
    x_window = x_start
    y_window = y_end - k
    # число итераций:
    x_iter = int((x_end - x_start) / k)
    y_iter = int((y_end - y_start) / k)
    # выборка центров линий:
    center_sample = get_center(lines_sample)
    center_x_sample = center_sample[:, 0]
    center_y_sample = center_sample[:, 1]

    #     windows = []  # для отрисовки окон (необяз)
    rho = []
    for i in range(y_iter):
        for j in range(x_iter):
            idxs1 = np.where((center_x_sample > x_window) & (center_x_sample < (x_window + k)))[0]
            idxs2 = np.where((center_y_sample > y_window) & (center_y_sample < (y_window + k)))[0]
            indexes = np.intersect1d(idxs1, idxs2)
            rho.append(len(indexes))

            # windows.append([x_window, y_window, x_window+k, y_window+k])  # для отрисовки окон (необяз)
            x_window += k

        x_window = x_start
        y_window -= k

    rho = np.array(rho)
    rho = rho / (k ** 2)

    # return rho, x_iter, y_iter
    rho_2d = rho.reshape((y_iter, x_iter))
    return rho_2d




