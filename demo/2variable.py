import torch
import numpy
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
# requires_grad = True 表示数据可以反向传播
variable = Variable(tensor, requires_grad = True)

t_out = torch.mean(tensor*tensor) # X^2
v_out = torch.mean(variable*variable)
print("\nt_out", t_out)
print("\nv_out", v_out)
# varibale 的反向传播
v_out.backward()
#  v_out = 1/4 * sum(var*var)
# d(v_out)/d(var) = 1/4 * 2var = 1/2 var
# 查看variable的梯度
print("\ngradient", variable.grad)
# 查看variable
print("\nvariable", variable)
# 查看variable中自变量
print("\nvariable_data", variable.data)
# 参数转化为numpy形式
print(variable.data.numpy())