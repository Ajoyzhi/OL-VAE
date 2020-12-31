"""
data = [] 创建为是numpy中的list
可以通过np.array(data)转化为array
无论是numpy的list或者array，均需要通过torch.tensor()/torch.FloatTensor()/from_numpy()转化为tensor
torch中的方法均需要tensor类型
"""
import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
    "\nnumpy_data", np_data,
    "\ntorch_data", torch_data,
    "\ntensor2array", tensor2array
)

# ab
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) # 32bitfloat
# sin;cos;mean
print(
    "\nabs",
    "\nnumpy", np.abs(data),
    "\ntensor", torch.abs(tensor)
)

# matrix
data = [[1, 2],[3, 4]] # list
tensor = torch.FloatTensor(data)
data_array = np.array(data)
print(
    "\nmul",
    "\nnumpy", np.matmul(data, data), # = np.dot()
    "\nnumpy_dot", data_array.dot(data_array),
    "\ntensor", torch.mm(tensor, tensor),# != torch.dot()
    # "\ntensor_dot", tensor.dot(tensor) # pytorch3.0之后对dot方法进行更新，只能对一维数据进行操作
)


