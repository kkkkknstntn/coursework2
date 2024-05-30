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
        sum_edj = m * (m - 1) + h * (l * (l - 1 + 2 * m))
        p = [i / sum_edj for i in repeated_nodes2]
        for i in range(source, source + l):
            targets = np.random.choice(repeated_nodes1, m, p=p, replace=False)
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
            if ylabel == r'$количество вершин$':
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


def friendship_paradox(G):
    sum_ind = 0
    n = len(G)
    for i in range(n):
        a_i = average_neighbor_degree(G, i)
        sum_ind += a_i / len(G[i])
    # print(sum_ind / n)
    return sum_ind / n


def graphs(n, k, m, l, metrics, cluster_bool=False):
    mx = max(metrics)
    deg_fin = np.zeros(n * l + m)
    deg_dic_fin = {}

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
        deg_dic = {}
        deg_dic_help = {}
        for g in range(len(G)):
            deg_fin[g] += len(G[g])
            if cluster_bool:
                deg_dic.setdefault(len(G[g]), 0)
                deg_dic_help.setdefault(len(G[g]), 0)
                deg_dic[len(G[g])] += local_cluster(G, g)
                # print(local_cluster(G, g))
                deg_dic_help[len(G[g])] += 1

        if cluster_bool:
            for deg in deg_dic:
                deg_dic_fin.setdefault(deg, 0)
                deg_dic_fin[deg] += deg_dic[deg] / deg_dic_help[deg]

                # print(deg_dic_fin[deg])
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
    if cluster_bool:
        for deg in deg_dic_fin:
            deg_dic_fin[deg] /= k
        deg_dic_fin = {k: deg_dic_fin[k] for k in sorted(deg_dic_fin)}
        dic = (np.array(list(deg_dic_fin.keys())), np.array(list(deg_dic_fin.values())))

        write_array_to_csv(dic[0], f'clusters{m}{l}.csv', 'w')
        write_array_to_csv(dic[1], f'clusters{m}{l}.csv', 'a')



        plots([dic],
              r'$d_i$', r'$с(d_i)$', labels=["m = {}, l = {}".format(m, l)])
        plots([dic],
              r'$d_i$', r'$с(d_i)$', labels=["m = {}, l = {}".format(m, l)], log=True)
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)])
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)], log=True)
    for name, param in zip(["d", "neig_deg", "fr_ind"], [r'$d_i$', r'$\alpha_i$', r'$\beta_i$']):
        mass = [(d, dict_metr[name + str(i)]) for i in metrics]
        plots(mass, r'$t$', param, labels=labels)
        plots(mass, r'$t$', param, labels=labels, log=True)


def local_cluster(G, i):
    d = dict(zip(G[i], np.zeros(len(G[i]))))
    for j in G[i]:
        for k in G[j]:
            if k in d:
                d[k] += 1
    return sum(d.values()) / ((len(G[i]) - 1) * len(G[i]))


def cluster(G):
    G = [np.array(g) for g in G]
    arr = np.empty(len(G), dtype=float)
    for h in range(len(G)):
        arr[h] = local_cluster(G, h)
        # print(np.mean(np.array(arr)))
    return np.mean(np.array(arr))


def cluster_graph(n, k, arr_m, arr_l):
    keys = np.array(arr_m)
    results = {}
    results_fr = {}
    for i in arr_l:
        results[i] = np.zeros(len(keys))
    for i in arr_m:
        results_fr[i] = np.zeros(len(arr_l))
    for h in range(k):
        print(h)
        for i in range(len(keys)):
            for key in results:
                results[key][i] += cluster(barabasi_albert_graph(keys[i], key, n)[0])
                # results[key][i] += friendship_paradox(barabasi_albert_graph(keys[i], key, n)[0])
        for i in range(len(arr_l)):
            for key in results_fr:
                results_fr[key][i] += friendship_paradox(barabasi_albert_graph(key, arr_l[i], n)[0])
    matrix1, matrix2, labels, labels_m = [], [], [], []
    for l in results:
        results[l] /= k
        matrix1.append((keys, results[l]))
        labels.append("l = {}".format(l))
    for m in results_fr:
        results_fr[m] /= k
        matrix2.append((np.array(arr_l), results_fr[m]))
        labels_m.append("m = {}".format(m))

    data_to_write1=""

    for k in range(len(matrix1[0][0])):
        data_to_write1 += str(matrix1[0][0][k])
        for i in range(len(matrix1)):
            data_to_write1 += (" " + str(matrix1[i][1][k]))
        data_to_write1+="\n"
    with open(f'graphs_fin/clusters.txt', mode='w') as txt_file:
        txt_file.write(data_to_write1)

    data_to_write1 = ""

    for k in range(len(matrix2[0][0])):
        data_to_write1 += str(matrix2[0][0][k])
        for i in range(len(matrix2)):
            data_to_write1 += (" " + str(matrix2[i][1][k]))
        data_to_write1 += "\n"
    with open(f'graphs_fin/friendship.txt', mode='w') as txt_file:
        txt_file.write(data_to_write1)

    plots(matrix1, r'$m$', r'$\overline{c}$', labels=labels)
    plots(matrix2, r'$l$', r'$Средний индекс дружбы$', labels=labels_m)


def main():
    # cluster_graph(1000, 10, [2, 5, 10, 25], [1, 3, 5, 10, 25])
    graphs(10000, 100, 5, 3, [10], True)


if __name__ == "__main__":
    main()
