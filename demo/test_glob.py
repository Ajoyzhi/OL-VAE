from other.path import pro_root
import glob
switch_num = 1
for j in range(switch_num):
    i = j + 1
    switch_path = pro_root + str(i) + "/*.csv"
    path_file_num = glob.glob(switch_path)
    print("switch:", i,
          "csv_file:", path_file_num,
          "csv_num:", len(path_file_num))