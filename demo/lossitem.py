import  torch
import numpy as np
# 一个batch_loss
tensor_loss = torch.tensor([[1.0,2.0,3.0,4.0],[2.0,3.0,4.0,5.0],[3.0,4.0,5.0,6.0]])
loss = [ ]
# loss相当于2个batch
loss.append(tensor_loss)
loss.append(tensor_loss)

# 测试batch求平均
mu = []
for item in loss:
    sum = torch.mean(item, dim=0)
    mu.append(sum)

# 测试list求平均函数
def list_avrg(list):
    sum = 0
    for item in list:
        sum += item

    return sum / len(list)

final = list_avrg(mu)
print(final)

# 测试tensor与numpy
numpy_loss = tensor_loss.numpy()
for item in numpy_loss:
    print(item)

# 测试矩阵乘法
a = torch.from_numpy(np.arange(6).reshape(2,3))
e = torch.from_numpy(np.arange(6).reshape(3,2))
b = a * a
c = a.mm(e)
d = a.mul(a)

print(
    "a:", a,
    "b:", b,
    "c:", c,
    "d:", d
)

#测试对角矩阵生成
a = torch.tensor([1.0, 2.0, 3.0])
diag = np.diag(a)
print(
    "a:", a,
    "diag:", diag
)

# 测试in {}
for i in range(5):
    if i in {1,2,3}:
        print(i)