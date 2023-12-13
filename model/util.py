import numpy as np
import pandas as pd
import random

def one_hot_decoding_predicted(data):

    new = []
    for e in data:
        node_label = []
        for node in e:
            node_label.append(np.argmax(node))
        new.append(node_label)

    new = np.array(new).flatten()
    return new

def top_k_rows(data, k):

    row_sum = []
    for i in range(len(data)):
        row_sum.append([np.sum(data[i]), i])

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[0])
    row_sum = row_sum[:k]

    row_sum = [e[1] for e in row_sum]
    random.seed(1)

    return np.array(row_sum)

def split_graph(data, k, split):

    graph = []

    if data.ndim == 2:
        for i in range(1, split+1):
            matrix = data[k*(i - 1) : k*i, k*(i - 1) : k*i]
            graph.append(matrix)
    else:
        for i in range(1, split+1):
            matrix = data[k*(i - 1) : k*i]
            graph.append(matrix)

    return np.array(graph)