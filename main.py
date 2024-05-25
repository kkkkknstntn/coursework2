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


def dict_def(n):
    return {
        "d10": np.zeros(n),
        "d50": np.zeros(n),
        "d100": np.zeros(n),
        "fr_ind10": np.zeros(n),
        "fr_ind50": np.zeros(n),
        "fr_ind100": np.zeros(n),
        "neig_deg10": np.zeros(n),
        "neig_deg50": np.zeros(n),
        "neig_deg100": np.zeros(n)
    }


def barabasi_albert_graph(m, l, n):
    G = complete_graph(n * l + m, m)
    dict = dict_def(n - 100)
    repeated_nodes1 = [i for i in range(m)]
    repeated_nodes2 = [len(G[i]) for i in range(m)]
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
        if h >= 100:
            h2 = h - 100
            for i in [10, 50, 100]:
                dict["d" + str(i)][h2] = len(G[i * l + m - 1])
                dict["neig_deg" + str(i)][h2] = average_neighbor_degree(G, i * l + m - 1)
                dict["fr_ind" + str(i)][h2] = dict["neig_deg" + str(i)][h2] / dict["d" + str(i)][h2]
    return G, dict


def plots(m, xlabel, ylabel,
          colors=['r', 'g', 'b'],
          colors2=['c', 'm', 'y'],
          styles=['-', '--', ':'],
          labels=['i =10', 'i =50', 'i =100'],
          log=False,
          ):
    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    flag = True
    for i, data in enumerate(m):
        x, y = data
        color, style, label = colors[i], styles[i], labels[i]
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
                plt.plot(x, y, color=color, linestyle=style, label=label)
                flag = False
                lbl = f'$y ={a:.1f}log(d_i) {b:+.1f}$'
            plt.plot(xseq, yseq, color=colors2[i], label=lbl)
        if flag:
            plt.plot(x, y, color=color, linestyle=style, label=label)

    plt.legend()
    plt.show()



def graphs(n, k, m, l):
    deg_fin = np.zeros(n * l + m)
    d = np.arange(100, n)
    dict = dict_def(n - 100)

    def write_array_to_csv(array, filename, s):
        with open(filename, s, newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(array)

    def write_arrays_to_csv(dict, s, arr=None):
        for key in dict:
            filename = f'{key}.csv'
            with open(filename, s, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if arr is None:
                    writer.writerow(dict[key])

                else:
                    writer.writerow(arr)

    write_arrays_to_csv(dict, 'w', d)

    for i in range(k):
        print(i)
        G, dict_2 = barabasi_albert_graph(m, l, n)
        for g in range(len(G)):
            deg_fin[g] += len(G[g])

        for key in dict:
            dict[key] += dict_2[key]
        write_arrays_to_csv(dict_2, 'a')

    for key in dict:
        dict[key] /= k
    deg_fin /= k

    write_arrays_to_csv(dict, 'a')
    deg_uniq1 = np.unique(deg_fin, return_counts=True)
    write_array_to_csv(deg_uniq1[0], f'degrees.csv', 'w')
    write_array_to_csv(deg_uniq1[1], f'degrees.csv', 'a')

    labels = ["i =10, m = {}, l = {}".format(m, l),
              "i =50, m = {}, l = {}".format(m, l),
              "i =100, m = {}, l = {}".format(m, l)
              ]
    cluster(G)
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)])
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)], log=True)

    for name, param in zip(["d", "neig_deg", "fr_ind"], [r'$d_i$', r'$\alpha_i$', r'$\beta_i$']):
        plots([(d, dict[name + "10"]), (d, dict[name + "50"]), (d, dict[name + "100"])],
              r'$t$', param, labels=labels)
        plots([(d, dict[name + "10"]), (d, dict[name + "50"]), (d, dict[name + "100"])],
              r'$t$', param, labels=labels, log=True)


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


def cluster_graph(n, k, arr):
    keys = np.array(arr)
    results = {
        1: np.zeros(len(keys)),
        3: np.zeros(len(keys)),
        5: np.zeros(len(keys))
    }
    for h in range(k):
        print(h)
        for i in range(len(keys)):
            for key in results:
                results[key][i] += cluster(barabasi_albert_graph(keys[i], key, n)[0])
    matrix = []
    lables = []
    for l in results:
        results[l] /= k
        matrix.append((keys, results[l]))
        lables.append("l = {}".format(l))
    plots(matrix, r'$m$', r'$\overline{c}$', labels=lables)


def main():
    # cluster_graph(1000, 10, [2, 5, 10, 25])
    graphs(1000, 10, 3, 3)


if __name__ == "__main__":
    main()
