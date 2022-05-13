from pandas import DataFrame
from sklearn.datasets import make_blobs, make_moons, make_circles, make_regression
import math
import csv
import numpy as np
import matplotlib.pyplot as plt


def blobs():
    X, y = make_blobs(n_samples=20, centers=2, n_features=2)
    X = np.around(X, decimals=3)
    df = DataFrame(dict(x1=X[:, 0], x2=X[:, 1], label=y))
    df.to_csv('./dataset/blobs_20.csv', sep=',', index=False, header=True)
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x1',
                   y='x2', label=key, color=colors[key])
    plt.show()


def moons():
    X, y = make_moons(n_samples=500, noise=0.1)
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    df.to_csv('./dataset/moons.csv', sep=',', index=False, header=True)
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x',
                   y='y', label=key, color=colors[key])
    plt.show()


def circles():
    X, y = make_circles(n_samples=500, noise=0.05)
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    df.to_csv('./dataset/circles.csv', sep=',', index=False, header=True)
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x',
                   y='y', label=key, color=colors[key])
    plt.show()


def regression():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.2)
    plt.scatter(X, y)
    plt.show()


blobs()