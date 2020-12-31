import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

BATCH_SIZE = 100
EPOCH = 5

# 随机1000个生成二维数组，服从标准正态分布
x = torch.normal(torch.zeros(1000, 2))
# 只有一类数据
y = torch.ones(*x.size())

# 生成数据集
torch_dataset = Data.TensorDataset(x, y)
# 加载数据集
loader = Data.DataLoader(
    # 加载数据的数据集
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    # 每次采样是否打乱顺序
    shuffle=False,
    # 子进程的数量
    # 如果子进程数大于0，说明要进行多线程编程(一定要有一个主函数)
    num_workers=0
)

# plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c = 'b')
# plt.show()

'''
    定义网络结构
    一个输入层，一个隐藏层，一个输出层
'''
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

# 创建神经网络
net = Net(2, 10, 2)
# print(myNet)

# 定义优化方法
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 定义损失函数
def loss(data):
    # 求数据列的平均值
    mean = torch.mean(data, dim=0, keepdim=True)
    data_z = data - mean
    # 矩阵的对应位相乘
    loss_z = torch.mul(data_z, data_z)
    # 求每行数据的和
    loss_row_z = torch.sum(loss_z, dim=1, keepdim=True)
    # 求损失函数
    loss = loss_row_z.mean()
    return loss

# 打开交互模式，可以在一张图上动态显示
plt.ion()
plt.show()

'''
# 画x的图
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
plt.text(2, -3, 'BATCH_SIZE=%d' %BATCH_SIZE, fontdict = {'size' : 10, 'color' : 'green'})
# plt.show()
'''

for epoch in range(EPOCH):
    batch_loss = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        # 训练
        optimizer.zero_grad()
        # 一个batch的输出
        out = net(batch_x)
        # 计算损失函数
        torch_loss = loss(out)
        batch_loss += torch_loss
        # 反向传播
        torch_loss.backward()
        # 优化
        optimizer.step()
    # 输出每个epoch的损失
    print(
        'epoch:', epoch,
        #  '\nbatch_x:', batch_x,
        #  '\nout:', out,
        '\nloss:', batch_loss,
    )
   #  if epoch % (EPOCH / 10) == 0:
        # 可视化
    plt.scatter(out.data.numpy()[:, 0], out.data.numpy()[:, 1])
    # plt.text(0, 0, 'loss=%.4f' % torch_loss, fontdict={'size': 20, 'color': 'red'})
    plt.pause(1)

# 关闭交互模式
plt.ioff()
plt.show()


