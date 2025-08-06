import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

torch.manual_seed(1)

N_SAMPLES = 25  # 样本数量（训练集和测试集各20个样本）
N_HIDDEN = 300  # 神经网络隐藏层神经元数量（控制模型复杂度）

# ---------------------- 生成训练数据 ----------------------
# 生成输入特征x：在[-1, 1]范围内生成20个等间隔点，形状从(25,)变为(25, 1)（适合神经网络输入）
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# 生成标签y：构造线性关系 y = x + 噪声（模拟真实数据中的误差）
# torch.normal(mean, std)：生成正态分布噪声，均值为0，标准差为1
y = x + 0.4 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# ---------------------- 生成测试数据 ----------------------
# 生成测试集输入特征test_x：与训练集x分布相同（[-1, 1]等间隔25点）
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# 生成测试集标签test_y：与训练集y同分布（y = x + 噪声），但噪声独立于训练集
test_y = test_x + 0.4 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))


net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron屏蔽50%
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

#优化器优化
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()
#训练500次
for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:  # 每 10 步画一次图
        # 将神经网络转换成测试形式, 画好图之后改回 训练形式
        net_overfitting.eval()
        # Dropout网络切换到评估模式（核心！此时会关闭Dropout，使用所有神经元）
        net_dropped.eval()  # 因为 drop 网络在 train 的时候和 test 的时候参数不一样.
        ...
        # plotting
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        # 过拟合网络的测试集预测曲线
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        # Dropout网络的测试集预测曲线
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        plt.pause(0.1)
        ...
        # 将两个网络改回 训练形式
        net_overfitting.train()
        net_dropped.train()
plt.ioff()
plt.show()
