import csv
data_header = ['datapath', 'in_port', 'dst_mac', 'out_port', 'packet_count', 'byte_count']
data = [[1, 1, 1, 2, 10, 14],
        [1, 1, 2, 3, 4, 15]]
datafile = open('data.csv', 'w', newline='')
spamwriter = csv.writer(datafile, dialect='excel')
spamwriter.writerow(data_header)
for item in data:
    spamwriter.writerow(item)# 没有空行
datafile.close()