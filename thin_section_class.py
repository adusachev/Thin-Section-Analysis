
from functions.get_segments_sample import *
from functions.get_general_features import *
from functions.get_lineaments_sample import *
from functions.rose_histogram import rose_diagram_part
from functions.I_X_Y_classification import *
from functions.shannon_entropy import *
from functions.fractal_dimension import *
from functions.statistical_analysis import *
from functions.distributions_pdfs import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.1)
import os




class ThinSection:
    """
    Класс ThinSection используется для хранения информации о трещинах в шлифах горных пород
    Класс взаимодействует с shape файлами, состоящими из линеаментов, соответствующих трещинам в образце
    Атрибуты и методы класса предоставляют информацию о распределениях некоторых характеристик сети трещин

    Для создания экземпляра класса используются функции read_shape_file, get_length_sample,
    get_azimuth_sample_rad из модулей пакета functions

    Attributes
    ----------
     segments_sample : list or np.array
        выборка сегментов трещин, где каждый сегмент хранится в виде (x_start, y_start, x_end, y_end)
        координаты задаются в миллиметрах

    segments_sample_size : int
        размер выборки сегментов трещин

    segments_length_sample : list or np.array
        выборка длин сегментов (мм)

    name : str
        название шлифа

    tectonic_type : str
        тектонический тип шлифа

    path : str
        полный путь до shape файла

    azimuth_sample_rad : list or np.array
        выборка азимутов трещин в радианах

    azimuth_sample_deg : list or np.array
        выборка азимутов трещин в градусах

    window_size : float
        размер окна сканирования, которым считалась плотность трещин

    density_sample : list or np.array
        двумерное распределение плотности трещин в образце

    lines : list
        список вершин линеаментовб где каждый линеамент задан как список его вершин
        // e.g. [(x1, y1), (x2, y2), (x3, y3)])

    lineaments_sample : list
        список линеаментов шлифа, где каждый линеамент задан как список сегментов этого линеамента
        // e.g. np.array([[x1, y1, x2, y2], [x2, y2, x3, y3]])

    lineaments_length_sample : np.array
        выборка длин линеаментов (мм)

    lineaments_azimuth_sample : np.array
        выдорка азимутов линеаментов

    intersections_counts : np.array
        выборка числа пересечений линеаментов

    epsilon : float
        погрешность выделения линеаментов (в пикселях)

    XIY_coordinates : np.array
        координаты всех вершин типов X, I, Y
        // первый столбец - x, второй столбец - y, третий столбец - тип вершины (X: 0, Y: 1, I: 2)

    XIY_counts : list
        список количества вершин X, I, Ys

    X_nodes : int
        число вершин типа X

    I_nodes : int
        число вершин типа I

    Y_nodes : int
        число вершин типа Y

    XIY_counts_normed : list
        список количества вершин X, I, Y нормированных на их сумму

    entropy_L_az : float
        энтропия Шеннона совместного распределения длин линеаментов и их азимутов

    entropy_rho_az : float
        взвешенная энтропия Шеннона распределения азимутов с учетом пространственного распределения их плотности

    pixels_coords : np.array
        координаты пикселей линеаментов

    frac_dim : float
        фрактальная размерность сети трещин

    stat_tests_lineaments_length : pd.DataFrame
        таблица с результатами критерия отношения правдоподобия

    powerlaw_alpha_length : float
        оценка максимального правдоподобия параметра степенного распеделения

    exponential_lambda_length : float
        оценка максимального правдоподобия параметра экспоненциального распеделения

    lognormal_mu_length : float
        оценка максимального правдоподобия среднего лог-нормального распеделения

    lognormal_sigma_length : float
        оценка максимального правдоподобия стандартного отклонения лог-нормального распеделения

    """

    def __init__(self, path, name=None, tectonic_type=None,
                 left=-np.inf, right=np.inf, bottom=-np.inf, top=np.inf):
        # segment features
        self.segments_sample = read_shape_file(path,
                                               left, right, bottom, top)
        self.segments_sample_size = len(self.segments_sample)
        self.segments_length_sample = get_length_sample(self.segments_sample)

        # general features
        self.name = name
        self.tectonic_type = tectonic_type
        self.path = path
        self.azimuth_sample_rad = get_azimuth_sample_rad(self.segments_sample)
        self.azimuth_sample_deg = self.azimuth_sample_rad * (180 / np.pi)
        self.window_size = choose_window_size(self.segments_length_sample)
        self.density_sample = compute_density(self.segments_sample, k=self.window_size)
        self.lines = get_lines(path)  # new

        # lineament features
        self.lineaments_sample = get_lines_segments_sample(self.lines)  # new
        self.lineaments_sample_size = len(self.lineaments_sample)
        self.lineaments_length_sample = pixels_to_mm(trace_length_sample(self.lineaments_sample))  # new
        self.lineaments_azimuth_sample = get_lineament_azimuth_sample(self.lineaments_sample)  # new
        # self.lineaments_intersections_counts = None  # new

        # topological features
        self.intersections_counts = None
        self.epsilon = None
        self.XIY_coordinates = None
        self.XIY_counts = None
        self.X_nodes = None  # new
        self.I_nodes = None  # new
        self.Y_nodes = None  # new
        self.XIY_counts_normed = None  # new

        # entropy features
        self.entropy_L_az = None  # new
        self.entropy_rho_az = None  # new

        # fractal dimension features
        self.pixels_coords = None  # new
        self.frac_dim = None  # new

        # statistical features
        # new (start)
        self.stat_tests_lineaments_length = None
        self.powerlaw_alpha_length = None
        self.exponential_lambda_length = None
        self.lognormal_mu_length = None
        self.lognormal_sigma_length = None

        self.stat_tests_lineaments_inter = None
        self.powerlaw_alpha_inter = None
        self.exponential_lambda_inter = None
        self.lognormal_mu_inter = None
        self.lognormal_sigma_inter = None
        # new (end)




    def perform_IXY_classification(self, epsilon):
        """
        Выполняет топологическую классификацию вершин трещин и точек пересечения трещин на типы I, X, Y

        :param epsilon: погрешность определения точки как Y-точки
        :return: None
        """
        # new
        XIY_nodes_coords, XIY_counts, intersections_counts = classificate_fracture_network(self.lineaments_sample,
                                                                                           epsilon)

        self.XIY_coordinates = XIY_nodes_coords  # первый столбец - x, второй столбец - y, третий столбец - тип вершины (X - 0, Y - 1, I - 2)
        self.XIY_counts = XIY_counts
        self.X_nodes = XIY_counts[0]  # new
        self.I_nodes = XIY_counts[1]  # new
        self.Y_nodes = XIY_counts[2]  # new
        self.XIY_counts_normed = norm_array(XIY_counts)  # new
        self.epsilon = epsilon
        self.intersections_counts = intersections_counts  # new


    def calc_entropy(self):  # new
        """
        Вычисляет энтропию Шеннона:
        1) по линеаментам и азимутам; 2) по плотности и азимутам
        :return: None
        """
        self.entropy_L_az = shannon_2d(self.lineaments_length_sample,
                                       self.lineaments_azimuth_sample)

        self.entropy_rho_az = rho_and_az_entropy(self)


    def calc_fractal_dimension(self):  # new
        """
        Вычисляет фракталььную размерность сети трещин
        :return: None
        """
        self.pixels_coords = lines_to_pixels(self.lines)
        image = get_image(self.pixels_coords)
        self.frac_dim = compute_frac_dim(image)


    def perform_stat_analysis(self):
        """
        Выполняет статистический анализ сети трещин:
        1) приближение степенным, лог-нормальным, экспоненциальным распределниями
        выборок длин линеаментов и числа пересечений линеаментов
        2) сравнение распределений с помощбю критерия отношения правдоподобия
        :return: None
        """
        # lineaments lengths
        length_sample_truncated = truncate_sample(self.lineaments_length_sample)
        fit_length, table_length = perform_stat_tests(length_sample_truncated)
        self.stat_tests_lineaments_length = table_length
        self.powerlaw_alpha_length = fit_length.alpha
        self.exponential_lambda_length = fit_length.exponential.Lambda
        self.lognormal_mu_length = fit_length.lognormal.mu
        self.lognormal_sigma_length = fit_length.lognormal.sigma

        inter_sample_truncated = truncate_sample(self.intersections_counts,
                                                 bins=len(np.unique(self.intersections_counts)))
        fit_inter, table_inter = perform_stat_tests(inter_sample_truncated)
        self.stat_tests_lineaments_inter = table_inter
        self.powerlaw_alpha_inter = fit_inter.alpha
        self.exponential_lambda_inter = fit_inter.exponential.Lambda
        self.lognormal_mu_inter = fit_inter.lognormal.mu
        self.lognormal_sigma_inter = fit_inter.lognormal.sigma



    def draw_length_MLE(self, figsize=(13, 6), bins=40, xscale='log', yscale='log', **kwargs):
        """
        Для выборки длин линеаментов рисует гистограмму обрезанной выборки
         и оценки максимального правдоподобия
        :param figsize: размер фигуры
        :param bins: число бинов гистограммы
        :param xscale: масштаб по оси x
        :param yscale: масштаб по оси y
        :param kwargs: kwargs для гистограммы
        :return: None
        """
        length_sample_truncated = truncate_sample(self.lineaments_length_sample)
        sample_sorted = np.sort(length_sample_truncated)
        xmin = np.min(sample_sorted)


        plt.figure(figsize=figsize)
        # приближенные распределения
        plt.plot(sample_sorted, exponential_pdf(sample_sorted, xmin, self.exponential_lambda_length),
                 color='C0', lw=3, label='exponential MLE')

        plt.plot(sample_sorted, lognormal_pdf(sample_sorted, xmin,
                                              self.lognormal_mu_length, self.lognormal_sigma_length),
                 color='C1', lw=3, label='lognormal MLE')

        plt.plot(sample_sorted, powerlaw_pdf(sample_sorted, xmin, self.powerlaw_alpha_length),
                 color='C2', lw=3, label='powerlaw MLE')

        # гистограмма
        plt.hist(sample_sorted, bins=bins, density=True, alpha=0.4, color='C3', label='data', **kwargs)

        plt.yscale(yscale)
        plt.xscale(xscale)
        plt.ylabel('Плотность вероятности')
        plt.xlabel('Длина, мм')
        plt.grid()
        plt.legend()
        # plt.show()

    def draw_intersections_MLE(self, figsize=(13, 6), xscale='log', yscale='log', **kwargs):
        """
        Для выборки числа пересечений линеаментов рисует гистограмму обрезанной выборки
         и оценки максимального правдоподобия
        :param figsize: размер фигуры
        :param bins: число бинов гистограммы
        :param xscale: масштаб по оси x
        :param yscale: масштаб по оси y
        :param kwargs: kwargs для гистограммы
        :return: None
        """
        bins = len(np.unique(self.intersections_counts))
        inter_sample_truncated = truncate_sample(self.intersections_counts, bins)
        sample_sorted = np.sort(inter_sample_truncated)
        sample_sorted = sample_sorted[np.where(sample_sorted > 0)[0]]  ## correction
        xmin = np.min(sample_sorted)

        plt.figure(figsize=figsize)
        # приближенные распределения
        plt.plot(sample_sorted, exponential_pdf(sample_sorted, xmin, self.exponential_lambda_inter),
                 color='C0', lw=3, label='exponential MLE')

        plt.plot(sample_sorted, lognormal_pdf(sample_sorted, xmin,
                                              self.lognormal_mu_inter, self.lognormal_sigma_inter),
                 color='C1', lw=3, label='lognormal MLE')

        plt.plot(sample_sorted, powerlaw_pdf(sample_sorted, xmin, self.powerlaw_alpha_inter),
                 color='C2', lw=3, label='powerlaw MLE')

        # гистограмма
        plt.hist(sample_sorted, bins=len(np.unique(sample_sorted)),
                 density=True, alpha=0.4, color='C3', label='data', **kwargs)

        plt.yscale(yscale)
        plt.xscale(xscale)
        plt.ylabel('Плотность вероятности')
        plt.xlabel('Число пересечений')
        plt.grid()
        plt.legend()
        # plt.show()



    def draw_segments_length_distribution(self, bins=50, xscale='linear', yscale='log',
                                          density=True, **kwargs):
        """Изображает гистограмму расределения длин трещин"""
        plt.figure(figsize=(8, 8))

        plt.hist(self.segments_length_sample, bins=bins, density=density, **kwargs)

        plt.xlabel('Длина сегмента трещины, мм')
        plt.ylabel('Плотность вероятности')
        plt.xscale(xscale)
        plt.yscale(yscale)
        # plt.show()


    def draw_trace_length_distribution(self, bins=50, xscale='linear', yscale='log',
                                          density=True, **kwargs):
        """Изображает гистограмму расределения длин трещин"""
        plt.figure(figsize=(8, 8))

        plt.hist(self.lineaments_length_sample, bins=bins, density=density, **kwargs)

        plt.xlabel('Длина трещины, мм')
        plt.ylabel('Плотность вероятности')
        plt.xscale(xscale)
        plt.yscale(yscale)
        # plt.show()


    def draw_azimuth_distribution(self, bins=36, xscale='linear', yscale='linear', density=True, **kwargs):
        """Изображает гистограмму расределения азимутов трещин"""
        plt.figure(figsize=(8, 8))

        plt.hist(self.azimuth_sample_deg, bins=bins, density=density, **kwargs)

        plt.xlabel(r'Угол наклона трещины, $~^{o}$')
        plt.ylabel('Плотность вероятности')
        plt.xscale(xscale)
        plt.yscale(yscale)
        # plt.show()


    def draw_rose_diagram_segments(self, bins=36):
        """
        Изображает роз-диаграмму
        Для корректной работы необходимо четное количество бинов
        """
        ax = rose_diagram_part(self.azimuth_sample_deg, bins=(bins / 2), min_edge=0, max_edge=180)

        rose_diagram_part(self.azimuth_sample_deg + 180, bins=(bins / 2), min_edge=180, max_edge=360, ax=ax)
        # plt.show()


    def draw_rose_diagram_lineaments(self, bins=36):
        """
        Изображает роз-диаграмму
        Для корректной работы необходимо четное количество бинов
        """
        ax = rose_diagram_part(self.lineaments_azimuth_sample, bins=(bins / 2), min_edge=0, max_edge=180)

        rose_diagram_part(self.lineaments_azimuth_sample + 180, bins=(bins / 2), min_edge=180, max_edge=360, ax=ax)
        # plt.show()


    def update_density_sample(self, window_size=None):
        """
        Вычисление распределения плотности через функцию compute_density

        :param window_size: размер окна сканирования в функции compute_density
        :return: None
        """
        if window_size is None:
            window_size = choose_window_size(self.segments_length_sample)

        self.window_size = window_size
        self.density_sample = compute_density(self.segments_sample, k=self.window_size)




    def draw_density_histogram_1d(self, window_size=None, bins=10, density=True, kde=False, **kwargs):
        """
        Рисует гистограмму распределения плотности трещин

        :param bins: количество бинов
        :param window_size: размер окна сканирования (см get_density_sample)
        :return: None
        """
        if window_size is not None:
            self.update_density_sample(window_size=window_size)

        m, k = self.density_sample.shape
        rho_1d = self.density_sample.reshape(m * k)

        plt.figure(figsize=(8, 8))

        if kde:
            density = True
            sns.kdeplot(rho_1d, color='r', lw=3)

        plt.hist(rho_1d, bins=bins, density=density, **kwargs)

        plt.xlabel(r'Плотность трещин, $\frac{N}{mm^2}$')
        plt.ylabel('Плотность вероятности')
        plt.title('Распределение плотности трещин')
        # plt.show()


    def draw_density_histogram_2d(self, window_size=None, cmap="YlGnBu", save_path=None, **kwargs):
        """
        Рисует двумерную тепловую карту распределения плотности трещин

        :param window_size: размер окна сканирования (см get_density_sample)
        :return: None
        """
        if window_size is not None:
            self.update_density_sample(window_size=window_size)

        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(self.density_sample, cmap=cmap, **kwargs)
        plt.title(r'Распределение плотности трещин, $\frac{N}{mm^2}$')
        plt.xticks([])
        plt.yticks([])
        plt.show()

        if save_path is not None:
            fig.savefig(save_path)




    def draw_intersections_histogram(self, xscale='linear', yscale='log', **kwargs):
        """
        Рисует гистограмму распределения числа пересечений всех трещин выборки
        Выдает ошибку, если выборка пересечений не подсчитано или не загружено
        :param bins: число бинов
                    (если None, то выбирается по числу уникальных значений числа пересечений)
        :param xscale: масштаб по оси x
        :param yscale: масштаб по оси y
        :param density: плотность вероятности по оси y
        :return: None
        """
        assert self.intersections_counts is not None, 'intersections sample is empty, run perform_IXY_classification() method'

        plt.figure(figsize=(13, 6))
        plt.hist(self.intersections_counts,
                 bins=len(np.unique(self.intersections_counts)), density=True, **kwargs)

        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlabel('Число пересечений')
        plt.ylabel('Плотность вероятности')
        plt.xticks(np.unique(self.intersections_counts))
        # plt.show()




