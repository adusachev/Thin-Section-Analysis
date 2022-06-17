
import numpy as np
import pandas as pd

import scipy.stats as sps
import powerlaw
from scipy.special import erfc


def truncate_sample(sample, bins=100, left_bound=None, right_bound=None):
    """
    По гистограмме обрезает выборку длин от значения максимального бина
    до первого бина, содержащего нулевое значение элементов выборки
    (или по вручную указанным границам)

    :param sample: выборка
    :param bins: число бинов гистограммы
    :param left_bound: левая граница для обрезки
    :param right_bound: правая граница для обрезки
    :return: обрезанная выборка
    """
    hist, values = np.histogram(sample, bins=bins, density=True)

    if left_bound is None:
        max_index = np.argmax(hist)  # индекс максимального зачения в бине гистограммы
        left_bound = values[max_index]

    if right_bound is None:
        zero_index_list = np.where(hist == 0)[0]  # список бинов с нулевым значением
        if len(zero_index_list) > 0:
            first_zero_index = zero_index_list[0]  # индекс первого бина с нулевым значением
            right_bound = values[first_zero_index]
        else:
            right_bound = np.max(sample)

    indexes = np.where((sample >= left_bound) & (sample <= right_bound))

    return sample[indexes]


def fit_distribution(sample, x_min=None):
    """
    Приближает выборку степенным, лог-нормальным и экспоненциальным распределениями
    Использует библиотеку powerlaw

    :param sample: выборка
    :param x_min: минимальное значение рассматриваемой величины,
                  начиная с которого выборка начинает описываться данным распределением
    :return: объект класса powerlaw.Fit - приближение выборки различными распределениями
    """
    if x_min is None:
        x_min = np.min(sample)

    fit = powerlaw.Fit(sample, xmin=x_min)
    return fit



def loglikelihood_ratio_test(sample, distr1, distr2, normalized_ratio=False):
    """
    Критерий отношения правдоподобия

    :param sample: выборка, которую приближают распределениями distr1 и distr2
    :param distr1: первое тестируемое распределение,
                    (distr1 должно иметь метод distr1.pdf(x),
                     вычисляющий плотность для точки или массива)
    :param distr2: альтернативное распределение,
                    (distr2 должно иметь метод distr2.pdf(x),
                     вычисляющий плотность для точки или массива)
    :param normalized_ratio: если True, возвращает нормализованное значение отношения правдоподобия
    :retrun: R, p
             (R - логарифмическое отношение правдоподобия,
              p - p-value)
    """
    n = len(sample)
    p1 = distr1.pdf(sample)
    p2 = distr2.pdf(sample)
    p2[np.where(p2 == 0)[0]] = 1e-10

    R_p = np.log(p1 / p2)
    R = np.sum(R_p)

    sigma = np.std(R_p)
    # p-value:
    p = erfc(np.abs(R) / np.sqrt(2 * n * (sigma ** 2)))

    if normalized_ratio:
        R = R / (sigma * np.sqrt(n))

    return R, p


def get_results_table(sample, fit, normalized_ratio=True):
    """
    Заносит результаты критериев отношения правдоподобия в таблицу

    :param sample: выборка
    :param fit: приближение выборки (объект класса powerlaw.Fit)
    :param normalized_ratio: если True, возвращает нормализованное значение отношения правдоподобия
    :return: таблица результатов теста отношения правдоподобия (pd.DataFrame)
    """
    columns = ['powerlaw vs. exponential',
               'powerlaw vs. lognormal',
               'lognormal vs. exponential']
    compared_distributions = np.array([['powerlaw', 'exponential'],
                                       ['powerlaw', 'lognormal'],
                                       ['lognormal', 'exponential']])

    test_results = np.array(
        [loglikelihood_ratio_test(sample, fit.power_law, fit.exponential, normalized_ratio=normalized_ratio),
         loglikelihood_ratio_test(sample, fit.power_law, fit.lognormal, normalized_ratio=normalized_ratio),
         loglikelihood_ratio_test(sample, fit.lognormal, fit.exponential, normalized_ratio=normalized_ratio)])
    ratios = test_results[:, 0]
    pvals = test_results[:, 1]

    greater_zero = (np.array(ratios) < 0).astype('int')

    fisrt_hyp = compared_distributions[0, greater_zero[0]]
    second_hyp = compared_distributions[1, greater_zero[1]]
    third_hyp = compared_distributions[2, greater_zero[2]]

    better_fit = [fisrt_hyp, second_hyp, third_hyp]

    significant = np.array(pvals) < 0.1

    # test_results = list(np.round_(ratios, decimals=3).T)
    pvals = list(np.round_(pvals, decimals=3).T)

    table_dict = {'R': ratios, 'p': pvals,
                  'better fit': better_fit,
                  'significant': significant}

    table = pd.DataFrame(table_dict, index=columns)
    return table




def perform_stat_tests(sample):
    """
    Выполняет шаги статистического анализа
    :param sample: выборка
    :return: 1) fit - приближение выборки распределениями (объект powerlaw.Fit)
             2) table - таблица с результатами критерия отношения правдоподбия
    """
    # перевод из пикселей в миллиметры
    #     length = scale_shapefile_elements(data.lineaments_length_sample)
    #     length = data.segments_length_sample
    # избавляемся от случайных нулей в выборке и обрезаем ее
    # length = length[np.where(length != 0)[0]]
    # sample = truncate_sample(length)

    # избавляемся от нулевых и отрицательных значений
    sample = sample[np.where(sample > 0)[0]]

    # приближение методом макс правдоподобия
    fit = fit_distribution(sample)

    table = get_results_table(sample, fit, normalized_ratio=True)
    return fit, table










