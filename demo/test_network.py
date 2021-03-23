from torch.autograd import Variable
import torch

x = Variable(torch.randn(2,2))
y = Variable(torch.randn(2,2))
z = Variable(torch.randn(2,2), requires_grad=True)


a = x+y
b = a+z

print(x.requires_grad, y.requires_grad, z.requires_grad) # False, False, True
print(a.requires_grad, b.requires_grad) # False, True

print(x.requires_grad) # False
print(a.requires_grad) # False

print("###################################################")

a = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
print("a:", a)
b = a + 3
print("b:", b)
c = b * b * 3
print("c:", c)
out = c.mean()
print(
    "out:", out,
    "dtype:", out.dtype,
)
out.backward(retain_graph=True) # 这里可以不带参数，默认值为‘1’，由于下面我们还要求导，故加上retain_graph=True选项
# 其实是函数out对a的导数
print(
    "gradient of a", a.grad.data,
    "dtype:", a.grad.dtype,
) # tensor([15., 18.])

print("#####################################################")
# 生成计算图
x = Variable(torch.ones([1]), requires_grad=True)
y = 0.5 * (x + 1).pow(2)
z = torch.log(y)

# 进行backward
# 注意这里等价于z.backward(torch.Tensor([1.0]))，参数表示的是后面的输出对Variable z的梯度
z.backward()
print(x.grad)

# 此时y.grad为None，因为backward()只求图中叶子的梯度（即无父节点）,如果需要对y求梯度，则可以使用`register_hook`
#print(y.grad)