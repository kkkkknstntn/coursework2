import matplotlib.pyplot as plt
import numpy as np
import csv


def complete_graph(n, m):
    return [[j for j in range(m) if j != i] if i < m else [] for i in range(n)]


def average_neighbor_degree(G, n):
    neighbor_degrees_sum = sum(len(G[neighbor]) for neighbor in G[n])
    num_neighbors = len(G[n])
    average_degree = neighbor_degrees_sum / num_neighbors if num_neighbors > 0 else 0
    return average_degree


def dict_def(n, arr):
    s = ["d", "fr_ind", "neig_deg"]
    return {j + str(i): np.zeros(n) for i in arr for j in s}


def barabasi_albert_graph(m, l, n, metr=None):
    G = complete_graph(n * l + m, m)
    repeated_nodes1 = [i for i in range(m)]
    repeated_nodes2 = [len(G[i]) for i in range(m)]
    mx = n
    dict_metr = {}
    if metr is not None:
        mx = max(metr)
        dict_metr = dict_def(n - mx, metr)
    for h in range(n):
        source = h * l + m
        new_repeated_nodes = []
        s = m * (m - 1) + h * (l * (l - 1 + 2 * m))
        sm = [i / s for i in repeated_nodes2]
        for i in range(source, source + l):
            targets = np.random.choice(repeated_nodes1, m, p=sm, replace=False)
            for target in targets:
                G[target].append(i)
                G[i].append(target)
            new_repeated_nodes.extend(targets)
            for j in range(i + 1, source + l):
                G[i].append(j)
                G[j].append(i)
        for i in new_repeated_nodes:
            repeated_nodes2[i] += 1
        for i in range(source, source + l):
            repeated_nodes2.append(m + l - 1)
            repeated_nodes1.append(i)
        if h >= mx:
            h2 = h - mx
            for i in metr:
                dict_metr["d" + str(i)][h2] = len(G[i * l + m - 1])
                dict_metr["neig_deg" + str(i)][h2] = average_neighbor_degree(G, i * l + m - 1)
                dict_metr["fr_ind" + str(i)][h2] = dict_metr["neig_deg" + str(i)][h2] / dict_metr["d" + str(i)][h2]
    return G, dict_metr


def plots(m, xlabel, ylabel,
          styles=['-', '--', ':', '-.', 'dashdot'],
          labels=['', '', '', '', ''],
          log=False,
          ):
    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    flag = True
    for i, data in enumerate(m):
        x, y = data
        style, label = styles[i], labels[i]
        if log:
            x = np.log(x)
            y = np.log(y)
            b, a = np.polyfit(x, y, deg=1)
            xseq = np.linspace(x.min(), x.max(), num=100)
            yseq = a + b * xseq
            lbl = f'$y = {a:.1f}log({xlabel}) {b:+.1f}$'
            if len(m) == 1:
                first_negative_index = np.where(yseq >= 0)[0][-1]
                xseq = xseq[:first_negative_index]
                yseq = yseq[:first_negative_index]
                plt.plot(x, y, linestyle=style, label=label)
                flag = False
                lbl = f'$y ={a:.1f}log(d_i) {b:+.1f}$'
            plt.plot(xseq, yseq, label=lbl)
        if flag:
            plt.plot(x, y, linestyle=style, label=label)

    plt.legend()
    plt.show()


def graphs(n, k, m, l, metrics):
    mx = max(metrics)
    deg_fin = np.zeros(n * l + m)
    d = np.arange(mx, n)
    dict_metr = dict_def(n - mx, metrics)

    def write_array_to_csv(array, filename, s):
        with open(filename, s, newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(array)

    def write_arrays_to_csv(dict_metr, s, arr=None):
        for metr in dict_metr:
            filename = f'{metr}.csv'
            with open(filename, s, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if arr is None:
                    writer.writerow(dict_metr[metr])

                else:
                    writer.writerow(arr)

    write_arrays_to_csv(dict_metr, 'w', d)
    for i in range(k):
        print(i)
        G, dict_2 = barabasi_albert_graph(m, l, n, metrics)
        for g in range(len(G)):
            deg_fin[g] += len(G[g])
        for key in dict_metr:
            dict_metr[key] += dict_2[key]
        write_arrays_to_csv(dict_2, 'a')
    for key in dict_metr:
        dict_metr[key] /= k
    deg_fin /= k
    write_arrays_to_csv(dict_metr, 'a')
    deg_uniq1 = np.unique(deg_fin, return_counts=True)
    write_array_to_csv(deg_uniq1[0], f'degrees.csv', 'w')
    write_array_to_csv(deg_uniq1[1], f'degrees.csv', 'a')
    labels = ["i ={}, m = {}, l = {}".format(i, m, l) for i in metrics]
    cluster(G)
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)])
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)], log=True)
    for name, param in zip(["d", "neig_deg", "fr_ind"], [r'$d_i$', r'$\alpha_i$', r'$\beta_i$']):
        mass = [(d, dict_metr[name + str(i)]) for i in metrics]
        plots(mass, r'$t$', param, labels=labels)
        plots(mass, r'$t$', param, labels=labels, log=True)


def cluster(G):
    G = [np.array(g) for g in G]
    arr = np.empty(len(G), dtype=float)
    for h, mas in enumerate(G):
        d = dict(zip(mas, np.zeros(len(mas))))
        for i in mas:
            for j in G[i]:
                if j in d:
                    d[j] += 1
        arr[h] = sum(d.values()) / ((len(mas) - 1) * len(mas))
    return np.mean(np.array(arr))


def cluster_graph(n, k, arr_m, arr_l):
    keys = np.array(arr_m)
    results = {}
    for i in arr_l:
        results[i] = np.zeros(len(keys))
    for h in range(k):
        print(h)
        for i in range(len(keys)):
            for key in results:
                results[key][i] += cluster(barabasi_albert_graph(keys[i], key, n)[0])
    matrix, labels = [], []
    for l in results:
        results[l] /= k
        matrix.append((keys, results[l]))
        labels.append("l = {}".format(l))
    plots(matrix, r'$m$', r'$\overline{c}$', labels=labels)


def main():
    cluster_graph(1000, 100, [2, 5, 10, 25], [1, 3,  5])
    # graphs(1000, 5, 3, 3, [10, 50, 100])


if __name__ == "__main__":
    main()
