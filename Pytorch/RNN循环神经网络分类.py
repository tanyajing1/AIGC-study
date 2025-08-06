import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision
import matplotlib
matplotlib.use('TkAgg')

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28      # rnn 时间步数 / 图片高度
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
LR = 0.01           # learning rate学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    #ToTensor()：将图片（PIL 格式）转换为 PyTorch 张量，并自动归一化到 0-1。
    transform=torchvision.transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    download=DOWNLOAD_MNIST,
)
#测试集
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_x = test_x.squeeze(1)
test_y = test_data.targets[:2000]  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)

#构建RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     # LSTM 效果要比 nn.RNN() 好多了
            input_size=28,      # 图片每行的数据像素点，输入特征维度
            hidden_size=64,     # 隐藏层神经元数量
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10) # 输出层 全连接层输出10类（数字0-9）

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x)   # None 表示 hidden state 会用全0的 state
        # r_out: [batch_size, 28, 64] (所有时间步的输出)
        # h_n: [1, batch_size, 64] (最后一个时间步的隐藏状态)
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # 使用Adam优化器
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数
for epoch in range(EPOCH):  # 遍历所有训练轮次
    for step, (x, b_y) in enumerate(train_loader):  # 遍历数据加载器
        # 重塑为RNN输入格式
        b_x = x.squeeze(1)  # reshape x to (batch, time_step, input_size)
        # 前向计算 b_x → LSTM层 → 取最后时间步 → 全连接层
        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)   # 计算损失
        optimizer.zero_grad()  # 清空梯度缓存
        loss.backward()  # 反向计算梯度
        optimizer.step()  # 更新参数

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print('Epoch:', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')