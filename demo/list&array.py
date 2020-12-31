import numpy as np

test_data = [[1, 2, 3],
             [4, 5, 6]]
print("二维数据:", test_data)


x1 = list(np.array(test_data).T[0:2])
print("第一列数据:", x1)

x2 = list(np.array(test_data).T[1])
print("第二列数据:", x2)

x3 =  list(np.array(test_data).T[2])
print("第三列数据:",x3)

x1.append(x2)
x1.append(x3)

print("New X1:", x1)