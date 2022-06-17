import numpy as np
from numba import njit
from numba_progress import ProgressBar
import numpy_indexed as npi
import warnings
warnings.filterwarnings("ignore")


def setdiff_2d(array1, array2):
    """
    Аналог np.setdiff1d для двумерных массивов
    :param array1: первый массив, shape = (n, 2)
    :param array2: второй массив, shape = (m, 2)
    :return: numpy-массив элементов array1, которых нет в array2
    """
    array1_set = set([tuple(elem) for elem in array1])
    array2_set = set([tuple(elem) for elem in array2])

    return np.array(list(array1_set - array2_set))


"""
Intersection fact and intersection point
"""


@njit
def rotate(A, B, C):
    """
    Определяет, с какой стороны от отрезка лежит заданная точка
    :param A: (a_x, a_y) - начало отрезка
    :param B: (b_x, b_y) - конец отрезка
    :param C: (c_x, c_y) - точка
    """
    a_x, a_y = A
    b_x, b_y = B
    c_x, c_y = C

    det = a_x * (b_y - c_y) - a_y * (b_x - c_x) + 1 * (b_x * c_y - b_y * c_x)

    return det


@njit
def intersection_criteria(A, B, C, D):
    """
    Критерий пересечения двух отрезков
    :param A: (a_x, a_y) - начало 1-го отрезка
    :param B: (b_x, b_y) - конец 1-го отрезка
    :param C: (c_x, c_y) - начало 2-го отрезка
    :param D: (d_x, d_y) - конец 2-го отрезка
    :return: True, если отрезки пересекаются
    """
    requirement_1 = rotate(A, B, C) * rotate(A, B, D) <= 0
    requirement_2 = rotate(C, D, A) * rotate(C, D, B) <= 0
    ans = requirement_1 and requirement_2

    return ans


@njit
def solve_2d_system(vec_a, vec_b, vec_c, delta):
    """
    Метод Крамера решения системы 2-х уравнений
    Cистема: vec_c = alpha * vec_a + beta * vec_b

    :param vec_a: массив коэффициентов, np.array([x_a, y_a])
    :param vec_b: массив коэффициентов, np.array([x_b, y_b])
    :param vec_c: массив свободных членов, np.array([x_c, y_c])
    :param delta: ранее вычисленный детерминант матрицы коэффициетнов, det([[x_a, x_b], [y_a, y_b]])
    :return: alpha, beta - неизвестные переменнвые системы
    """
    #     delta = vec_a[0] * vec_b[1] - vec_b[0] * vec_a[1]  # вычислен в intersection
    delta_1 = vec_c[0] * vec_b[1] - vec_b[0] * vec_c[1]
    delta_2 = vec_a[0] * vec_c[1] - vec_c[0] * vec_a[1]

    assert delta != 0, 'System is inconsistent'
    alpha = delta_1 / delta
    beta = delta_2 / delta

    return alpha, beta


@njit
def intersection(x1, y1, x2, y2,
                 x3, y3, x4, y4):
    """
    Вычисляет точку пересечения двух сегментов с помощью метода Крамера

    x1, y1, x2, y2 - координаты первого отрезка
    x3, y3, x4, y4 - координаты второго отрезка
    :return: np.array([x_inter, y_inter])
             None, если точки пересечения нет
    """
    vec_a = np.array([x2 - x1,
                      y2 - y1])
    vec_b = np.array([x4 - x3,
                      y4 - y3])
    vec_c = np.array([x3 - x1,
                      y3 - y1])
    # учет ошибки, возникающей, например, когда все точки двух сегментов имеют одну и ту же координату по x
    delta = vec_a[0] * vec_b[1] - vec_b[0] * vec_a[1]
    if delta == 0:
        return None

    alpha, beta = solve_2d_system(vec_a, vec_b, vec_c, delta)
    point_inter = np.array([x3, y3]) - beta * vec_b

    return point_inter


@njit
def get_intrersection_points(line1_segments, line2_segments):
    """
    Возвращает список точек пересечения двух трещин

    :param line1_segments: список сегментов первого линеамента  // e.g. np.array([[x1, y1, x2, y2],
                                                                                  [x2, y2, x3, y3]])
    :param line2_segments: список сегментов второго линеамента
    :return: список всех точек пересечения двух линеаментов
    """
    intersection_points = []

    for seg1 in line1_segments:
        for seg2 in line2_segments:
            # проверка на наличие пересечения в принципе
            if intersection_criteria(seg1[:2], seg1[2:],
                                     seg2[:2], seg2[2:]):
                # точка пересечения
                point = intersection(seg1[0], seg1[1], seg1[2], seg1[3],
                                     seg2[0], seg2[1], seg2[2], seg2[3])
                if point is not None:
                    intersection_points.append(point)

    return intersection_points


"""
The ability of two cracks intersecting
"""


@njit
def trace_center(line_segments):
    """
    Возвращает центр спрямленного линамента
    :param line_segments: cписок сегментов данного линеамента  // e.g. np.array([[x1, y1, x2, y2],
                                                                                  [x2, y2, x3, y3]])
    :return: (center_x, center_y) - центр спрямленного линамента
    """
    start_x, start_y = line_segments[0][:2]
    end_x, end_y = line_segments[-1][2:]
    center_x = (start_x + end_x) / 2
    center_y = (start_y + end_y) / 2

    return center_x, center_y


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


@njit
def able_to_intersect(line1_segments, line2_segments):
    """
    Критерий того, способны ли в принципе пересекаться линеаменты
    (вычисляет расстояние между центрами линеаментов и сравнивает его с их длинами;
     если расстояние не превышает определенный порог, то лиеаменты, в теории, могут пересекаться)

    :param line1_segments: cписок сегментов первого линеамента  // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]])
    :param line2_segments: cписок сегментов второго линеамента
    :return: True, если пересечение возможно, False - иначе
    """
    center1 = trace_center(line1_segments)
    center2 = trace_center(line2_segments)
    length1 = trace_length(line1_segments)
    length2 = trace_length(line2_segments)

    if dist(center1, center2) < 0.6 * (length1 + length2):
        return True

    return False



"""
X-Y classifier
"""


@njit
def get_vertexes(line_segments):
    """
    Возвращает начальную и конечную точки линеамента

    :param line_segments: cписок сегментов данного линеамента  // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]])
    :return: (x_start, y_start), (x_end, y_end)
    """
    start = line_segments[0][:2]
    end = line_segments[-1][-2:]

    return start, end


@njit
def classificate_point_XY(line1_segments, line2_segments, intersection_point, epsilon):
    """
    Классифицирует точку пересечения двух линеаментов как X или как Y
    Точке назначается класс Y в случае, если расстояние от нее до начала/конца
    одного из линеаментов не превышает погрешность epsilon,
    в противном случае точка будет иметь класс X

    :param line1_segments: список сегментов первого линеамента  // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]])
    :param line2_segments: список сегментов второго линеамента
    :param intersection_point: точка пересечения [x_inter, y_inter]
    :param epsilon: погрешность длины линеамента
    :return: класс точки - 'X' или 'Y'
    """
    start1, end1 = get_vertexes(line1_segments)
    start2, end2 = get_vertexes(line2_segments)

    node_type = 'X'

    if dist(intersection_point, start1) < epsilon:
        line1_segments[0][0] = intersection_point[0]
        line1_segments[0][1] = intersection_point[1]
        node_type = 'Y'

    if dist(intersection_point, end1) < epsilon:
        line1_segments[-1][-2] = intersection_point[0]
        line1_segments[-1][-1] = intersection_point[1]
        node_type = 'Y'

    if dist(intersection_point, start2) < epsilon:
        line2_segments[0][0] = intersection_point[0]
        line2_segments[0][1] = intersection_point[1]
        node_type = 'Y'

    if dist(intersection_point, end2) < epsilon:
        line2_segments[-1][-2] = intersection_point[0]
        line2_segments[-1][-1] = intersection_point[1]
        node_type = 'Y'

    return node_type


@njit
def prolong_segments(line_segments, epsilon):
    """
    Продлевает первый и последний сегменты данного линеамента на величину epsilon

    :param line_segments: cписок сегментов данного линеамента  // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]])
    :param epsilon: величина, на которую удлинняются концы линеамента (погрешность длины)
    :return: копия line_segments, с удлиненными концами,
             ИЛИ None, если случайно попался сегмент нулевой длины

    (P1, P2 - точки концов сегмента; Q1 - новый конец сегмента после удлиннения со стороны точки P1;
     Q2 - новый конец сегмента после удлиннения со стороны точки P2)
    """
    line_segments_tmp = np.copy(line_segments)

    # 1) удлинение начала
    x1, y1, x2, y2 = line_segments_tmp[0]
    P2P1_coords = np.array([x1 - x2, y1 - y2])  # вектор P1P2
    P2P1_length = dist((x1, y1), (x2, y2))

    if P2P1_length == 0:
        return None

    P2Q1_coords = ((P2P1_length + epsilon) / P2P1_length) * P2P1_coords
    x1_new, y1_new = P2Q1_coords + np.array([x2, y2])
    line_segments_tmp[0][0] = x1_new  # обновление
    line_segments_tmp[0][1] = y1_new

    # 2) удлинение конца
    x1, y1, x2, y2 = line_segments_tmp[-1]
    P1P2_coords = np.array([x2 - x1, y2 - y1])
    P1Q2_coords = ((P2P1_length + epsilon) / P2P1_length) * P1P2_coords
    x2_new, y2_new = P1Q2_coords + np.array([x1, y1])

    line_segments_tmp[-1][-2] = x2_new  # обновление
    line_segments_tmp[-1][-1] = y2_new

    return line_segments_tmp

# new (start)
@njit
def unique_elements(array):
    """
    Возвращает уникальные значения двуменого массива
    Преобразует элементы массива из np.array([x, y]) в tuple(x, y)
    :param array: двумерный массив (или список массивов размера 2)
    :return: список уникальных элементов исходного массива
                (тип: список кортежей размера 2)
    """
    array_unique = []
    for elem in array:
        elem = elem[0], elem[1]  # преобразование в tuple
        if elem not in array_unique:
            array_unique.append(elem)

    return array_unique
# new (end)


@njit(nogil=True)
def classificate_network_XY(lines_segments, progress_proxy, epsilon):
    """
    Классифицирует точки пересечения линеаментов X и Y

    :param lines_segments: список линеаментов, где каждый линеамент задается как cписок сегментов данного линеамента
    :param progress_proxy: вывод прогресса цикла в numba
    :param epsilon: погрешность длины линеамента
    :return: 1) all_inter_points - список координат всех найденных точек пересечения
             2) node_types - список типов точек пересечения
    """
    n = len(lines_segments)
    all_inter_points = []
    node_types = []

    intersections_counts = np.zeros(n)  # new

    for i in range(n):
        line1_segments = lines_segments[i]
        for j in range(i + 1, n):
            line2_segments = lines_segments[j]

            # проверка на возможность пересечения в принципе
            if able_to_intersect(line1_segments, line2_segments):
                # список точек пересечения для рассматриваемых 2-х трещин
                two_inters_list = get_intrersection_points(line1_segments, line2_segments)

                # локализация точек пересечения и классификация
                if len(two_inters_list) > 0:

                    for point in unique_elements(two_inters_list):  # new (added unique_elements)
                        all_inter_points.append(np.array(point))  # new (point -> np.array(point))
                        node_type = classificate_point_XY(line1_segments, line2_segments, point,
                                                          epsilon)
                        node_types.append(node_type)

                        ## new (start)
                        if node_type == 'X':
                            intersections_counts[i] += 1
                            intersections_counts[j] += 1
                        ## new (end)

                else:
                    line1_segments_tmp = prolong_segments(line1_segments, epsilon=epsilon)
                    line2_segments_tmp = prolong_segments(line2_segments, epsilon=epsilon)

                    # обработка случая сегментов нулевой длины
                    if (line1_segments_tmp is None) or (line2_segments_tmp is None):
                        continue
                    # список точек пересечения для рассматриваемых 2-х трещин
                    two_inters_list_2 = get_intrersection_points(line1_segments_tmp,
                                                                 line2_segments_tmp)
                    # локализация точек пересечения
                    if len(two_inters_list_2) > 0:
                        for point in two_inters_list_2:

                            node_type = classificate_point_XY(line1_segments, line2_segments,
                                                              point, epsilon)
                            if node_type != 'X':  # в этом блоке точки в принципе не могут иметь тип Х
                                node_types.append(node_type)
                                all_inter_points.append(point)
        progress_proxy.update(1)

    return all_inter_points, node_types, intersections_counts  # new



"""
I-classification and post-processing
"""


def classificate_I_nodes(lines_segments, all_inter_points):
    """
    Классфицирует все концы линеаментов как I-вершины при условии,
    что ранее эти эти вершины не были определены как Y-вершины

    :param lines_segments: список линеаментов, где каждый линеамент задается как cписок сегментов данного линеамента
    :param all_inter_points: список координат всех точек пересечения
    :return: I_nodes - двумерный массив координат I точек
    """
    # массив координат концов линеаментов
    vertexes = []
    for line_segments in lines_segments:
        start, end = get_vertexes(line_segments)
        vertexes.append(start)
        vertexes.append(end)
    vertexes = np.array(vertexes)

    I_nodes = setdiff_2d(vertexes, all_inter_points)
    return I_nodes



def classification_result(lines_segments, all_inter_points, node_types):
    """
    Итоговый результат I-X-Y классификации

    :param lines_segments: список линеаментов, где каждый линеамент задается как cписок сегментов данного линеамента
    :param all_inter_points: список координат всех точек пересечения
    :param node_types: список классов точек пересечения (X или Y)
    :return: X_sample, Y_sample, I_sample - двумерные массивы координат точек имеющих типы X, Y, I соответственно
    """
    all_inter_points = np.array(all_inter_points)
    node_types = np.array(node_types)

    indexes_x = np.where(node_types == 'X')[0]
    indexes_y = np.where(node_types == 'Y')[0]

    X_sample = all_inter_points[indexes_x]
    Y_sample = all_inter_points[indexes_y]
    I_sample = classificate_I_nodes(lines_segments, all_inter_points)

    X_sample = npi.unique(X_sample)
    Y_sample = npi.unique(Y_sample)
    I_sample = npi.unique(I_sample)  # по идее, не обязательно

    return X_sample, Y_sample, I_sample




# def main(lines_segments, epsilon):
#     n = len(lines_segments)
#
#     with ProgressBar(total=n) as progress:
#         all_inter_points, node_types = classificate_XY(lines_segments,
#                                                             progress,
#                                                             epsilon)
#     X_sample, Y_sample, I_sample = classification_result(lines_segments, all_inter_points, node_types)
#
#     return X_sample, Y_sample, I_sample



def classificate_fracture_network(lines_segments, epsilon):
        n = len(lines_segments)

        with ProgressBar(total=n) as progress:  # new
            all_inter_points, node_types, intersections_counts = classificate_network_XY(lines_segments,
                                                                                         progress,
                                                                                         epsilon)
        X_sample, Y_sample, I_sample = classification_result(lines_segments, all_inter_points, node_types)

        X_sample_with_type = np.zeros((len(X_sample), 3))
        Y_sample_with_type = np.zeros((len(Y_sample), 3))
        I_sample_with_type = np.zeros((len(I_sample), 3))

        X_sample_with_type[:, :2] = X_sample
        X_sample_with_type[:, 2] = 0
        Y_sample_with_type[:, :2] = Y_sample
        Y_sample_with_type[:, 2] = 1
        I_sample_with_type[:, :2] = I_sample
        I_sample_with_type[:, 2] = 2

        XIY_nodes_coords = np.vstack((X_sample_with_type, I_sample_with_type, Y_sample_with_type))
        XIY_counts = [len(X_sample), len(I_sample), len(Y_sample)]

        return XIY_nodes_coords, XIY_counts, intersections_counts  # new



def norm_array(sample):
    """
    Нормирует массив на его сумму (нужно для вычисления пропорций X, I, Y)
    :param sample: массив
    :return: нормированный массив
    """
    return sample / np.sum(sample)
