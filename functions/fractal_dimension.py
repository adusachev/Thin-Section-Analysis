import numpy as np
# import shapefile


# def get_lines(path):
#     """
#     :param path: путь до .shp файла
#     :return: выборка линеаментов
#                     (каждый линеамент задан как список его вершин,
#                      e.g. [(x1, y1), (x2, y2), (x3, y3)])
#     """
#     sf = shapefile.Reader(path)
#     shapes = sf.shapes()
#     n = len(shapes)
#
#     lines = []
#     for i in range(n):
#         lines.append(shapes[i].points)
#
#     return lines


def lines_to_pixels(lines):
    """
    Конвертирует выборку линеаментов в пиксели

    :param lines: выборка линеаментов
                    (каждый линеамент задан как список его вершин,
                     e.g. [(x1, y1), (x2, y2), (x3, y3)])
    :return: grid - массив координат пикселей, в 1 столбце которого расположены x, во 2 - y
    """
    grid = []
    n = len(lines)

    for i in range(n):
        line = np.array(lines[i])
        points_x = line[:, 0]
        points_y = line[:, 1]

        points_number = len(points_x)

        for j in range(points_number - 1):
            x1 = points_x[j]
            x2 = points_x[j + 1]

            y1 = points_y[j]
            y2 = points_y[j + 1]

            angle = (np.arctan((y1 - y2) / (x1 - x2)) * 180) / np.pi

            if x1 == x2:
                for m in range(int(min(y1, y2)), int(max(y1, y2))):
                    pixel_x = x1
                    pixel_y = m
                    grid.append([pixel_x, pixel_y])

            elif (45 < angle < 90) or (-90 < angle < -45):
                for k in range(int(min(y1, y2)), int(max(y1, y2))):
                    pixel_x = int(((-x2 * y1 + x1 * y2) + (x2 - x1) * k) / (y2 - y1))
                    pixel_y = k
                    grid.append([pixel_x, pixel_y])

            else:
                for n in range(int(min(x1, x2)), int(max(x1, x2))):
                    pixel_x = n
                    pixel_y = int(((x2 * y1 - x1 * y2) + (y2 - y1) * n) / (x2 - x1))
                    grid.append([pixel_x, pixel_y])

    grid = np.array(grid)
    return grid


def get_image(pixels):
    """
    Получает изображение из списка пикселей

    :param pixels: массив координат пикселей, в 1 столбце которого расположены X, во втором - Y
    :return: двумерный массив, заполненный единицами в ячейках соответствующих
             координатам пикселя и нулями в остальных местах
    """
    x = pixels[:, 0]
    y = pixels[:, 1]

    # перенос начала координат в 0
    x = x.astype('int') - int(np.min(x))
    y = y.astype('int') - int(np.min(y))

    # размеры изображения в пикселях
    n = np.max(x) + 1
    m = np.max(y) + 1

    image = np.zeros((n, m))
    for i in range(len(x)):
        x_coord = x[i]
        y_coord = y[i]
        image[x_coord][y_coord] = 1

    return image





def boxcount(Z, k):
    """
    Разбивает двумерный массив Z на ячейки размера k
    Считает число ненулевых ячеек

    :param Z: двумерный массив (предположительно из 0 и 1)
    :param k: размер ячейки
    """
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])


def compute_frac_dim(image):
    """
    Вычисляет фрактальную размерность изображения с помощью boxcount algorithm

    :param image: двумерный массив изображения, состоящий из 0 и 1 (1 задают контур фигуры)
    :return: фрактальная размерность фигуры на изображении
    """
    windows = np.round_(np.logspace(np.log(2), np.log(2000), 15, base=np.e)).astype(int)
    counts = np.zeros(len(windows))

    for i, k in enumerate(windows):
        N = boxcount(image, k)
        counts[i] = N

    coeffs = np.polyfit(np.log(1 / windows), np.log(counts), 1)
    f = coeffs[0]

    return f

    # отрисовка результата линейной регрессии
    # if draw:
    #     plt.scatter(np.log(1 / windows), np.log(counts))
    #
    #     grid = np.linspace(np.min(np.log(1 / windows)), np.max(np.log(1 / windows)), 100)
    #     plt.plot(grid, coeffs[0] * grid + coeffs[1], color='C1')
    #
    #     plt.xlabel('log(1 / window size)')
    #     plt.ylabel('N')
    #     plt.grid()
    #     plt.show()



