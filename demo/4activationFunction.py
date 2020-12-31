import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5,5,200) # 在-5到5之间取200个点,shape=(200,1s)
print('tensor.shape', x.shape)
x = Variable(x)
x_np = x.data.numpy()
print('variable.shape:', x.shape)
print('x_np.shape',x_np.shape)


# use activation function calculate y:
y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
# y_softplus = torch.softplus(x).data.numpy()

# create graph
plt.figure(1,figsize=(4,6))
plt.subplot(221)
plt.plot(x_np,y_relu, c='red', label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_tanh, c='red', label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

# plt.subplot(224)
# plt.plot(x_np,y_softplus, c='red', label='softplus')
# plt.ylim((-0.2,6))
# plt.legend(loc='best')

plt.show()