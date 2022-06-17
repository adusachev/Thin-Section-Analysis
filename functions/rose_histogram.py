import matplotlib.pyplot as plt
import numpy as np


def rose_diagram_part(sample, bins=20, min_edge=0, max_edge=180, ax=None):
    """
    Изображает роз-диаграмму

    :param sample: выборка углов в градусах
    :param bins: количество бинов
    :param min_edge: левая граница первого бина
    :param max_edge: правая граница последнего бина
    :param ax: оси, в которых будет нарисована диаграмма

    :return: ax - оси, в которых будет нарисована диаграмма
    """
    edge_width = (max_edge - min_edge) / bins  # ширина бина в градусах
    bin_edges = np.arange(min_edge, max_edge + edge_width, edge_width)  # края бинов

    hist, bin_edges = np.histogram(sample, bins=bin_edges)

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.axes(projection='polar')
        ax = plt.gca()

    bars_coors = np.deg2rad(np.linspace(min_edge + (edge_width / 2),
                                        max_edge - (edge_width / 2),
                                        hist.shape[0]))

    ax.bar(bars_coors, hist,
           width=np.deg2rad(edge_width), color='.8', edgecolor='k', alpha=1)

    radii = np.linspace(0, np.max(hist), 6, endpoint=False)  # радиальная сетка
    ax.set_rgrids(radii=radii, labels=[])
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    ax.set_title('Rose Diagram of the "Fault System"', y=1.10, fontsize=15)

    return ax



def rose_diagram(sample, bins=36):
    """
    Изображает роз-диаграмму
    Для корректной работы необходимо четное количество бинов

    :param sample: выборка углов в градусах
    """
    ax = rose_diagram_part(sample, bins=(bins / 2), min_edge=0, max_edge=180)

    rose_diagram_part(sample + 180, bins=(bins / 2), min_edge=180, max_edge=360, ax=ax)
    plt.show()
