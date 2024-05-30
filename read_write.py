import csv
from itertools import islice
import numpy as np


def dict_def(n, arr):
    s = ["d", "fr_ind", "neig_deg"]
    return {j + str(i): np.zeros(n) for i in arr for j in s}


# # Открываем CSV-файл для чтения
# with open('graphs_fin/33/d10.csv', mode='r') as csv_file:
#     # Создаем объект reader для чтения данных из CSV-файла
#     csv_reader = csv.reader(csv_file)
#
#     # Переходим к первой строке (индексация начинается с 0)
#     first_row = next(csv_reader)
#
#     # Переходим к 102-й строке
#
#     second_row = next(islice(csv.reader(csv_file), 100, None))
#
# data_to_write = "i d\n"
# for i in range(len(first_row)):
#     data_to_write += "{} {} \n".format(first_row[i], second_row[i])
# # Подготавливаем данные для записи в TXT-файл
#
# # Записываем данные в TXT-файл
# with open('d10.txt', mode='w') as txt_file:
#     txt_file.write(data_to_write)
#
# print("Данные успешно записаны в output.txt")


def read_arrays_from_csv(dict_metr):
    for dir in ["33", "35", "53", "55"]:
        print(dir)
        for metr in dict_metr:
            filename = f'graphs_fin/{dir}/{metr}.csv'
            with open(filename, "r") as csvfile:
                csv_reader = csv.reader(csvfile)
                first_row = np.array([float(i) for i in (next(csv_reader))])
                second_row = np.array([float(i) for i in (next(islice(csv.reader(csvfile), 100, None)))])
                data_to_write = f't logt {metr} log{metr}\n'
                lg1 = np.log(first_row)
                lg2 = np.log(second_row)
                print(dir, metr, ": ", np.polyfit(lg1, lg2, deg=1))
                for i in range(0, len(first_row), 3):
                    data_to_write += "{} {} {} {} \n".format(first_row[i], lg1[i], second_row[i], lg2[i])
                with open(f'graphs_fin/{dir}/{metr}.txt', mode='w') as txt_file:
                    txt_file.write(data_to_write)
        filename = f'graphs_fin/{dir}/degrees.csv'
        with (open(filename, "r") as csvfile):
            csv_reader = csv.reader(csvfile)
            first_row = np.array([float(i) for i in (next(csv_reader))])
            second_row = np.array([float(i) for i in (next(csv_reader))])
            lg1 = np.log(first_row)
            lg2 = np.log(second_row)
            x = [i for (i,j) in zip(lg1,lg2) if j >3]
            y = [j for (i, j) in zip(lg1, lg2) if j >3]
            print(dir, ": ", np.polyfit(x, y, deg=1))
            data_to_write = f't logt degr logdegr\n'
            for i in range(len(first_row)):

                data_to_write += "{} {} {} {} \n".format(first_row[i], lg1[i], second_row[i], lg2[i])
            with open(f'graphs_fin/{dir}/degrees.txt', mode='w') as txt_file:
                txt_file.write(data_to_write)


    for dir in ["50000_31", "50000_51", "30000_31", "30000_51"]:
        print(dir)
        for metr in dict_metr:
            filename = f'graphs_fin/{dir}/{metr}.csv'
            with open(filename, "r") as csvfile:
                csv_reader = csv.reader(csvfile)
                first_row = np.array([float(i) for i in (next(csv_reader))])
                second_row = np.array([float(i) for i in (next(islice(csv.reader(csvfile), 100, None)))])
                data_to_write = f't logt {metr} log{metr}\n'
                lg1 = np.log(first_row)
                lg2 = np.log(second_row)
                print(dir, metr, ": ", np.polyfit(lg1, lg2, deg=1))
                for i in range(0, len(first_row), 5):
                    data_to_write += "{} {} {} {} \n".format(first_row[i], lg1[i], second_row[i], lg2[i])
                with open(f'graphs_fin/{dir}/{metr}.txt', mode='w') as txt_file:
                    txt_file.write(data_to_write)
        filename = f'graphs_fin/{dir}/degrees.csv'
        with (open(filename, "r") as csvfile):
            csv_reader = csv.reader(csvfile)
            first_row = np.array([float(i) for i in (next(csv_reader))])
            second_row = np.array([float(i) for i in (next(csv_reader))])
            lg1 = np.log(first_row)
            lg2 = np.log(second_row)
            x = [i for (i, j) in zip(lg1, lg2) if j > 3]
            y = [j for (i, j) in zip(lg1, lg2) if j > 3]
            print(dir, ": ", np.polyfit(x, y, deg=1))
            data_to_write = f't logt degr logdegr\n'
            for i in range(0, len(first_row), 2):
                data_to_write += "{} {} {} {} \n".format(first_row[i], lg1[i], second_row[i], lg2[i])
            with open(f'graphs_fin/{dir}/degrees.txt', mode='w') as txt_file:
                txt_file.write(data_to_write)


dict = dict_def(10000 - 100, [10, 50, 100])

read_arrays_from_csv(dict)
