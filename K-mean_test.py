# -*-coding: utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import random

maximum = 500


def data_generation(num, radius=80):
    empty_set = []
    center = [[50, 100], [250, 300], [100, 300]]
    for c in center:
        for _ in range(num):
            r = random.random() * radius
            degree = random.random() * 2 * np.pi
            x = r * np.sin(degree)
            y = r * np.cos(degree)
            empty_set.append([x+c[0], y+c[1]])
    return empty_set, len(empty_set)


result = data_generation(100)
data_set = result[0]
labels = [None] * len(data_set)
index = []
for _ in range(result[1]):
    index.append(_)


def k_mean(centroid_num=3, iteration_num=250):
    initial_cent = random.sample(data_set, centroid_num)
    distance = []
    for _ in range(iteration_num):
        for _ in range(centroid_num):
            cent = initial_cent[_]
            distance.append([])
            for point in data_set:
                d = np.sqrt(np.square(cent[0]-point[0]) + np.square(cent[1]-point[1]))
                distance[_].append(d)
        for _ in range(len(data_set)):
            comparison = []
            for i in range(centroid_num):
                comparison.append(distance[i][_])
            label = comparison.index(max(comparison))
            labels[_] = label

        category = []
        for _ in range(centroid_num):
            category.append([])
        for _ in index:
            if labels[_] == 0:
                category[0].append(data_set[_])
            elif labels[_] == 1:
                category[1].append(data_set[_])
            else:
                category[2].append(data_set[_])

        for _ in range(len(category)):
            cluster = category[_]
            array = np.mean(cluster, 0)
            initial_cent[_][0] = array[0]
            initial_cent[_][1] = array[1]

    for _ in range(centroid_num):
        cent = initial_cent[_]
        distance.append([])
        for point in data_set:
            d = np.sqrt(np.square(cent[0]-point[0]) + np.square(cent[1]-point[1]))
            distance[_].append(d)
    for _ in range(len(data_set)):
        comparison = []
        for i in range(centroid_num):
            comparison.append(distance[i][_])
        label = comparison.index(max(comparison))
        labels[_] = label

    category = []
    for _ in range(centroid_num):
        category.append([])
    for _ in index:
        if labels[_] == 0:
            category[0].append(data_set[_])
        elif labels[_] == 1:
            category[1].append(data_set[_])
        else:
            category[2].append(data_set[_])

    plt.title('K-Mean Test')
    plt.xlim(-100, maximum+100)
    plt.ylim(-100, maximum+100)
    for _ in range(centroid_num):
        color = None
        if _ == 0:
            color = 'red'
        if _ == 1:
            color = 'green'
        if _ == 2:
            color = 'blue'
        for point in category[_]:
            plt.scatter(point[0], point[1], c=color, marker='o')
    for _ in initial_cent:
        plt.scatter(_[0], _[1], c='violet', marker='o')
        plt.annotate('Centroid', (_[0], _[1]))
    plt.show()


k_mean()
