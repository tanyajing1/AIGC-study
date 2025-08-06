import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 存放激励函数

# 创建数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 合并数据
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# 可视化原始数据
#plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()

# 定义神经网络
class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值
        return x

# 创建网络实例
net = Net(n_feature=2, n_hidden=10, n_output=2)  # 2个特征, 10个隐藏层神经元, 2个输出(两类)
print(net)  # 打印网络结构

# 定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数，适合分类问题

# 训练网络
for t in range(100):
    out = net(x)     # 喂给 net 训练数据 x, 输出分析值

    loss = loss_func(out, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

    # 每2次迭代可视化一次结果
    if t % 2 == 0:
        plt.cla()
        # 修正：明确指定softmax的维度为1，解决警告问题
        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 计算准确率
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()


#快速构建神经网络
# 原先创建网络实例
net = Net(n_feature=2, n_hidden=10, n_output=2)
#快速构建神经网络
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(), #激励函数层
    torch.nn.Linear(10, 2)
)