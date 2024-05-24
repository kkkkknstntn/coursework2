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


def barabasi_albert_graph(m, l, n):
    G = complete_graph(n * l + m, m)
    d10, d50, d100 = np.zeros(n - 100), np.zeros(n - 100), np.zeros(n - 100)
    fr_ind10, fr_ind50, fr_ind100 = np.zeros(n - 100), np.zeros(n - 100), np.zeros(n - 100)
    neig_deg10, neig_deg50, neig_deg100 = np.zeros(n - 100), np.zeros(n - 100), np.zeros(n - 100)
    n1 = n * l + m
    h = 0
    repeated_nodes1 = [i for i in range(m)]
    repeated_nodes2 = [len(G[i]) for i in range(m)]
    for source in range(m, n1, l):

        new_repeated_nodes = []
        s = m * (m - 1) + h * (l * (l - 1 + 2 * m))
        sm = [i / s for i in repeated_nodes2]

        for i in range(source, source + l):
            targets = np.random.choice(repeated_nodes1, m,
                                       p=sm, replace=False)

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

        if h >= 99:
            d10[h - 100] = len(G[10 * l + m - 1])
            d50[h - 100] = len(G[50 * l + m - 1])
            d100[h - 100] = len(G[100 * l + m - 1])
            neig_deg10[h - 100] = average_neighbor_degree(G, 10 * l + m - 1)
            neig_deg50[h - 100] = average_neighbor_degree(G, 50 * l + m - 1)
            neig_deg100[h - 100] = average_neighbor_degree(G, 100 * l + m - 1)
            fr_ind10[h - 100] = neig_deg10[h - 100] / d10[h - 100]
            fr_ind50[h - 100] = neig_deg50[h - 100] / d50[h - 100]
            fr_ind100[h - 100] = neig_deg100[h - 100] / d100[h - 100]
        h += 1
    return G, d10, d50, d100, fr_ind10, fr_ind50, fr_ind100, neig_deg10, neig_deg50, neig_deg100


def plots(m, xlabel, ylabel,
          colors=['r', 'g', 'b'],
          colors2=['c', 'm', 'y'],
          styles=['-', '--', ':'],
          labels=['i =10', 'i =50', 'i =100'],
          log=False,
          ):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(m)):
        x = m[i][0]
        y = m[i][1]
        if log:
            x = np.log(m[i][0])
            y = np.log(m[i][1])
            b, a = np.polyfit(x, y, deg=1)
            xseq = np.linspace(x[0], x[-1], num=100)
            yseq = a + b * xseq

            if len(m) == 1:
                first_negative_index = np.where(yseq < 0)[0][0]
                xseq = xseq[:first_negative_index]
                yseq = yseq[:first_negative_index]
                plt.plot(xseq, yseq, color=colors2[i], label=f'$y = {a:.1f}log(d_i) {b:+.1f}$')
            else:
                plt.plot(xseq, yseq, color=colors2[i], label=f'$y = {a:.1f}log({xlabel}) {b:+.1f}$')
        plt.plot(x, y, colors[i], linestyle=styles[i], label=labels[i])

    plt.legend()
    plt.show()


def graphs(n, k, m, l):
    d10, d50, d100 = np.zeros(n - 100), np.zeros(n - 100), np.zeros(n - 100)
    fr_ind10, fr_ind50, fr_ind100 = np.zeros(n - 100), np.zeros(n - 100), np.zeros(n - 100)
    neig_deg10, neig_deg50, neig_deg100 = np.zeros(n - 100), np.zeros(n - 100), np.zeros(n - 100)
    deg_fin = np.zeros(n * l + m)
    d = np.arange(100, n)

    def write_array_to_csv(array, filename, s):
        with open(filename, s, newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(array)

    write_array_to_csv(d, f'd10.csv', 'w')
    write_array_to_csv(d, f'd50.csv', 'w')
    write_array_to_csv(d, f'd100.csv', 'w')
    write_array_to_csv(d, f'fr_ind10.csv', 'w')
    write_array_to_csv(d, f'fr_ind50.csv', 'w')
    write_array_to_csv(d, f'fr_ind100.csv', 'w')
    write_array_to_csv(d, f'neig_deg10.csv', 'w')
    write_array_to_csv(d, f'neig_deg50.csv', 'w')
    write_array_to_csv(d, f'neig_deg100.csv', 'w')

    for i in range(k):
        print(i)
        G, d10_2, d50_2, d100_2, fr_ind10_2, fr_ind50_2, fr_ind100_2, neig_deg10_2, neig_deg50_2, neig_deg100_2 = \
            barabasi_albert_graph(m, l, n)

        for g in range(len(G)):
            deg_fin[g] += len(G[g])

        d10 += d10_2
        d50 += d50_2
        d100 += d100_2
        neig_deg10 += neig_deg10_2
        neig_deg50 += neig_deg50_2
        neig_deg100 += neig_deg100_2
        fr_ind10 += fr_ind10_2
        fr_ind50 += fr_ind50_2
        fr_ind100 += fr_ind100_2
        write_array_to_csv(d10_2, f'd10.csv', 'a')
        write_array_to_csv(d50_2, f'd50.csv', 'a')
        write_array_to_csv(d100_2, f'd100.csv', 'a')
        write_array_to_csv(fr_ind10_2, f'fr_ind10.csv', 'a')
        write_array_to_csv(fr_ind50_2, f'fr_ind50.csv', 'a')
        write_array_to_csv(fr_ind100_2, f'fr_ind100.csv', 'a')
        write_array_to_csv(neig_deg10_2, f'neig_deg10.csv', 'a')
        write_array_to_csv(neig_deg50_2, f'neig_deg50.csv', 'a')
        write_array_to_csv(neig_deg100_2, f'neig_deg100.csv', 'a')

    deg_fin /= k
    d10 /= k
    d50 /= k
    d100 /= k
    neig_deg10 /= k
    neig_deg50 /= k
    neig_deg100 /= k
    fr_ind10 /= k
    fr_ind50 /= k
    fr_ind100 /= k

    deg_uniq1 = np.unique(deg_fin, return_counts=True)
    write_array_to_csv(deg_uniq1[0], f'degrees.csv', 'w')
    write_array_to_csv(deg_uniq1[1], f'degrees.csv', 'a')
    # print(deg_uniq1)
    colors = ['r', 'g', 'b']
    styles = ['-', '--', ':']
    labels = ["i =10, m = {}, l = {}".format(m, l),
              "i =50, m = {}, l = {}".format(m, l),
              "i =100, m = {}, l = {}".format(m, l)
              ]
    cluster(G)
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)])
    plots([deg_uniq1],
          r'$d_i$', r'$количество вершин$', labels=["m = {}, l = {}".format(m, l)], log=True)

    plots([(d, d10), (d, d50), (d, d100)],
          r'$t$', r'$d_i$', labels=labels)
    plots([(d, d10), (d, d50), (d, d100)],
          r'$t$', r'$d_i$', labels=labels, log=True)

    plots([(d, neig_deg10), (d, neig_deg50), (d, neig_deg100)],
          r'$t$', r'$\alpha_i$', labels=labels)
    plots([(d, neig_deg10), (d, neig_deg50), (d, neig_deg100)],
          r'$t$', r'$\alpha_i$', labels=labels, log=True)

    plots([(d, fr_ind10), (d, fr_ind50), (d, fr_ind100)],
          r'$t$', r'$\beta_i$', labels=labels)
    plots([(d, fr_ind10), (d, fr_ind50), (d, fr_ind100)],
          r'$t$', r'$\beta_i$', labels=labels, log=True)


def cluster(G):
    for i in range(len(G)):
        G[i] = np.array(G[i])
    arr = []
    for mas in G:
        d = dict(zip(mas, np.zeros(len(mas))))
        for i in mas:
            for j in G[i]:
                if j in d:
                    d[j] += 1
        arr.append(sum(d.values()) / (len(mas) - 1) / (len(mas) - 2))
    return sum(arr) / len(arr)


def cluster_graph():
    d1 = {}
    d3 = {}
    d5 = {}
    for i in (3, 5, 10, 25):
        G1 = barabasi_albert_graph(i, 1, 1000)[0]
        d1[i] = cluster(G1)
        G3 = barabasi_albert_graph(i, 3, 1000)[0]
        d3[i] = cluster(G3)
        G5 = barabasi_albert_graph(i, 5, 1000)[0]
        d5[i] = cluster(G5)
    plt.figure()
    plt.plot(d1.keys(), d1.values())
    plt.plot(d3.keys(), d3.values())
    plt.plot(d5.keys(), d5.values())
    plt.legend()
    plt.show()




def main():
    # cluster_graph()
    graphs(1000, 2, 3, 5)


if __name__ == "__main__":
    main()
