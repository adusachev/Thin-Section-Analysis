import numpy as np
from scipy.special import erfc


def powerlaw_pdf(x, xmin, alpha):
    """
    Плотность степенного распределения
    """
    mult1 = (alpha - 1) / xmin
    mult2 = (x / xmin)**(-alpha)
    return mult1 * mult2


def exponential_pdf(x, xmin, Lambda):
    """
    Плотность экспоненциального распределения
    """
    mult1 = Lambda * np.exp(Lambda * xmin)
    mult2 = np.exp(- Lambda * x)
    return mult1 * mult2


def lognormal_pdf(x, xmin, mu, sigma):
    """
    Плотность лог-нормального распределния
    """
    mult1 = np.sqrt(2 / (np.pi * (sigma ** 2)))
    mult2 = 1 / erfc((np.log(xmin) - mu) / (sigma * np.sqrt(2)))
    mult3 = (1 / x) * np.exp(- ((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return mult1 * mult2 * mult3

