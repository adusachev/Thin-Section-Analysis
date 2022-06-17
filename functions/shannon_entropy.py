import numpy as np

from functions.get_general_features import length, get_length_sample, get_center, choose_window_size



def shannon_1d(sample, bins=18):
    """
    Энтропия Шеннона для одномерного распределения
    """
    hist = np.histogram(sample, bins=bins)[0]
    hist_normed = hist / np.sum(hist)

    S = np.nansum( - hist_normed * np.log2(hist_normed) )
    return S



def shannon_2d(sample1, sample2, bins=18):
    """
    Энтропия Шеннона совместного распределения двух величин
    """
    assert len(sample1) == len(sample2), 'Samples have different sizes'
    hist, _, _, = np.histogram2d(sample1, sample2, bins=bins)
    hist_normed = hist / np.sum(hist)
    S = np.nansum(- hist_normed * np.log2(hist_normed))

    return S



def weights_and_azimuths(data, k=None):
    """
    Едет по изображению ячейкой k x k
    Для каждой ячейки сохраняет выборку азимутов сегментов в ней
    Вычисляет вес ячейки как отношение числа сегментов в ней к общему числу сегментов в шлифе

    :param data: объект ThinSection (pickle файл)
    :param k: размер движущейся ячейки
    :return: 1) weights - список весов ячеек
             2) cell_az - список, элементами которого являются массивы азимутов сегментов каждой ячейки
                            (в cell_az лежат NaN, если в соотв ячейках нет ни одного сегмента)

    P.S. окно начинает движение с верхнего левого угла и едет вправо
        при постоянном y-ке до предела, потом опускается и тд
    """
    # выбор ширины окна (если не задано вручную)
    if k is None:
        # length_sample = get_length_sample(lines_sample)
        k = data.window_size

    lines_sample = data.segments_sample

    # локализация изображения
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

    rho = []
    cell_az = []
    for i in range(y_iter):
        for j in range(x_iter):
            idxs1 = np.where((center_x_sample > x_window) & (center_x_sample < (x_window + k)))[0]
            idxs2 = np.where((center_y_sample > y_window) & (center_y_sample < (y_window + k)))[0]
            indexes = np.intersect1d(idxs1, idxs2)
            rho.append(len(indexes))

            if len(indexes) > 0:
                cell_az.append(data.azimuth_sample_deg[indexes])
            else:
                cell_az.append(np.nan)

            x_window += k

        x_window = x_start
        y_window -= k

    rho = np.array(rho)
    weights = rho / len(lines_sample)  # теперь это не плотность а веса

    return weights, cell_az




def rho_and_az_entropy(data):
    """
    Вычисляет взвешенную энтропию Шеннона по плотности и азимутам

    :param data: объект ThinSection (pickle файл)
    :return: взвешенная энтропия
    """
    # вычисление весов и списков азимутов в каждй ячейке
    weights, cell_az = weights_and_azimuths(data)

    # энтропия в каждой ячейке
    s_list = []
    for az_sample in cell_az:
        if az_sample is np.nan:
            s_list.append(np.nan)
            continue
        s_list.append(shannon_1d(az_sample))

    # суммарная взвешенная энтропия
    S_result = np.nansum(s_list * weights)
    return S_result



