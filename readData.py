import re
import os
import pandas as pd
import numpy as np
import csv
import ast

label_define = ["base[0.95,(1,49)]","full-lastBest50%","full-genBest50","sub-lastBest50%","sub-genBest50"]

def get_file_num(file:str):
    f_num = filter(str.isdigit, file)
    f_list = list(f_num)
    f_str = []
    for i in range(len(f_list)):
        f_str.append("".join(f_list[i]))
    f_str = "".join(f_str)
    f_int = int(f_str)
    return f_int

def draw_plt(data):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("A test graph")
    x = [i for i in range(51)]
    for i in range(len(data)):
        y = data[i]
        plt.plot(x, y, label=label_define[i])
    plt.legend()
    plt.savefig(r'data/w_f.png', dpi=200)
    plt.show()

path = r'C:\Users\cjy\Desktop\TLdata'
files = os.listdir(path)
files_csv = list(filter(lambda x: x[-4:]=='.csv' , files))
all_data = []
data = []
for j in range(len(files)):
    file = files[j]
    with open(path + '/' + file) as f:
        # reader = csv.reader(f)
        # tmp = next(reader)
        tmp = f.readlines(1)
        # tmp = tuple(firstRow[0].strip('\n').split("\t"))
        # fieldnames = tuple(tmp[0].strip('\n').split("\t"))
        for i, value in enumerate(tmp):
            nValue = ast.literal_eval(value)
            tmp[i] = nValue
    if len(all_data) == 0:
        all_data = tmp[0]
    else:
        all_data = list(map(lambda x, y: x + y, all_data, tmp[0]))
    if (j+1) % 6 == 0:
        data.append(list(np.array(all_data) / 6))
        # print(data)
        all_data = []
draw_plt(data)




