import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main():
    # 设置随机种子保证结果可复现
    torch.manual_seed(1)

    # 超参数设置
    LR = 0.01  # 学习率(learning rate)
    BATCH_SIZE = 32  # 每批数据量
    EPOCH = 12  # 训练轮数

    # 创建模拟数据集
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # 生成1000个-1到1的等间距数，并增加维度
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))  # y = x^2 + 噪声

    # 绘制数据集散点图
    plt.scatter(x.numpy(), y.numpy())
    plt.show()

    # 准备数据加载器
    torch_dataset = Data.TensorDataset(x, y)  # 将数据包装成PyTorch Dataset
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 打乱数据顺序
        num_workers=2,  # 使用2个子进程加载数据
    )

    # 定义神经网络结构
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(1, 20)  # 隐藏层(输入1维，输出20维)
            self.predict = torch.nn.Linear(20, 1)  # 输出层(输入20维，输出1维)

        def forward(self, x):
            x = F.relu(self.hidden(x))  # 隐藏层使用ReLU激活函数
            x = self.predict(x)  # 输出层不使用激活函数(回归任务)
            return x

    # 创建4个相同的网络实例，用于比较不同优化器
    net_SGD = Net()  # 普通SGD
    net_Momentum = Net()  # 带动量的SGD
    net_RMSprop = Net()  # RMSprop
    net_Adam = Net()  # Adam
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # 创建对应的优化器
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)  # 添加动量
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)  # RMSprop
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))  # Adam
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    # 定义损失函数和记录器
    loss_func = torch.nn.MSELoss()  # 均方误差损失(回归任务常用)
    losses_his = [[], [], [], []]  # 记录4个网络的训练损失

    # 训练过程
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):  # 遍历数据加载器
            # 对每个网络/优化器组合进行训练
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)  # 前向传播
                loss = loss_func(output, b_y)  # 计算损失
                opt.zero_grad()  # 清空梯度(防止累积)
                loss.backward()  # 反向传播计算梯度
                opt.step()  # 更新参数
                l_his.append(loss.data.numpy())  # 记录损失值

    # 绘制结果
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    main()