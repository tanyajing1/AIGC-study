import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

torch.manual_seed(1)  # 设置随机种子保证可复现性

# 超参数设置
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  # 如果已经下载过MNIST可以设为False

# 下载MNIST数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    #ToTensor()：将图片（PIL 格式）转换为 PyTorch 张量，并自动归一化到 0-1。
    transform=torchvision.transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    download=DOWNLOAD_MNIST,
)

# 打印数据集信息
#print(train_data.data.size())  # 使用data而不是train_data
#print(train_data.targets.size())  # 使用targets而不是train_labels
# 显示第二张图片
#plt.imshow(train_data.data[1].numpy(), cmap='gray')  # 只显示第一张图片
#plt.title('Label: %i' % train_data.targets[1])
#plt.show()

#测试集
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)



# 定义卷积神经网络(CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Sequential(  # 输入形状 (1, 28, 28) [通道数, 高度, 宽度]
            nn.Conv2d(
                in_channels=1,  # 输入通道数(灰度图为1)
                out_channels=16,  # 输出通道数/卷积核数量
                kernel_size=5,  # 卷积核大小5x5
                stride=1,  # 卷积步长
                padding=2,  # 填充保证输出尺寸不变: padding=(kernel_size-1)/2 在图片周围一圈多加一圈0的数据，防止扫描出去了
            ),  # 输出形状 (16, 28, 28)
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2),  # 2x2最大池化，输出形状 (16, 14, 14) 选择2x2区域中最大的值，当作裁剪，高度不变
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(  # 输入形状 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 参数简写：输入16, 输出32, 核5x5, 步长1, 填充2
            nn.ReLU(),  # 输出形状 (32, 14, 14)
            nn.MaxPool2d(2),  # 输出形状 (32, 7, 7)
        )

        # 全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 将32x7x7的特征图展平，输出10个类别(数字0-9)

    def forward(self, x):
        # 前向传播
        x = self.conv1(x)  # 经过第一个卷积块
        x = self.conv2(x)  # 经过第二个卷积块
        x = x.view(x.size(0), -1)  # 展平多维特征图，保持batch_size维度
        output = self.out(x)  # 全连接层输出
        return output,x # 同时返回输出和展平特征

# 实例化网络
cnn = CNN()
#print(cnn)  # 打印网络结构

#训练模型
# 定义优化器和损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 使用Adam优化器优化CNN的所有参数
loss_func = nn.CrossEntropyLoss()  # 使用交叉熵损失函数（适用于分类问题）


# 可视化工具函数
def visualize_features(features, labels, title):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.cm as cm

        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=0)
        low_dim_features = tsne.fit_transform(features)

        # 绘制结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1],
                              c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(title)
        plt.show()

    except ImportError:
        print("请安装scikit-learn以使用t-SNE可视化: pip install scikit-learn")

plt.ion()  # 开启交互模式
# 训练循环
for epoch in range(EPOCH):  # 遍历所有epoch
    for step, (b_x, b_y) in enumerate(train_loader):  # 遍历训练数据加载器
        # b_x: 当前批次的图像数据，形状为[batch_size, 1, 28, 28]

        # b_y: 当前批次的标签，形状为[batch_size]

        output, _ = cnn(b_x)  # 只取预测输出，忽略特征  前向传播：将批次数据输入CNN网络，得到预测输出
        # output形状：[batch_size, 10]（10个类别的预测概率）
        loss = loss_func(output, b_y)  # 计算损失：预测输出和真实标签的交叉熵损失
        optimizer.zero_grad()  # 清空梯度缓存（重要！防止梯度累积）
        loss.backward()  # 反向传播：计算损失相对于所有参数的梯度
        optimizer.step()  # 参数更新：根据梯度更新网络权重
        if step % 100 == 0:
            # 测试和可视化
            cnn.eval()
            with torch.no_grad():
                test_output, features = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = (pred_y == test_y.numpy()).mean()
                print(f'Epoch: {epoch} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}')
                # 可视化特征
                if step % 500 == 0:  # 每500步可视化一次
                    visualize_features(features[:200].numpy(),
                                       test_y[:200].numpy(),
                                       f'Feature Space at Step {step}')
            cnn.train()
plt.ioff()  # 关闭交互模式

test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')