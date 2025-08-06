import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# 设置随机种子以保证实验可重复性
torch.manual_seed(1)  # 设置PyTorch的随机种子
np.random.seed(1)  # 设置NumPy的随机种子

# 超参数定义
BATCH_SIZE = 64  # 每批训练数据的大小
LR_G = 0.0001  # 生成器(Generator)的学习率
LR_D = 0.0001  # 判别器(Discriminator)的学习率
N_IDEAS = 5  # 生成器的输入噪声维度(可以理解为创意元素的个数)
ART_COMPONENTS = 15  # 每幅艺术作品由15个点组成(生成器输出的维度)

# 生成艺术作品的基准点(在-1到1之间等距生成15个点)
# 为BATCH_SIZE个作品生成相同的基准点(每行代表一个作品的基准点)
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


def artist_works():
    """
    模拟著名艺术家创作的真实艺术作品(真实样本)
    返回形状为[BATCH_SIZE, ART_COMPONENTS]的Tensor
    生成逻辑：
    1. 为每幅画生成一个随机系数a(范围1到2)
    2. 使用二次函数 y = a*x^2 + (a-1) 生成艺术作品
    3. 其中x是PAINT_POINTS中的基准点
    返回:
        paintings: 真实艺术作品，形状为[BATCH_SIZE, ART_COMPONENTS]的FloatTensor
    """
    # 为每幅画生成一个随机系数a(形状[BATCH_SIZE, 1])
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    # 使用二次函数生成艺术作品: y = a*x^2 + (a-1)
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    # 将NumPy数组转换为PyTorch FloatTensor
    paintings = torch.from_numpy(paintings).float()
    return paintings


# 生成器定义
G = nn.Sequential(
    # 第一层：将噪声向量(N_IDEAS维)映射到128维隐藏空间
    nn.Linear(N_IDEAS, 128),  # 全连接层，输入维度N_IDEAS(5)，输出维度128
    # 输入：随机噪声向量(可以来自正态分布)
    nn.ReLU(),  # ReLU激活函数，引入非线性
    # 第二层：将128维特征映射到艺术作品维度(ART_COMPONENTS=15)
    nn.Linear(128, ART_COMPONENTS),  # 全连接层，输入128，输出ART_COMPONENTS(15)
    # 输出：生成的艺术作品(15个点的y坐标)
)

# 判别器(Discriminator)网络定义
D = nn.Sequential(
    # 第一层：接收艺术作品(ART_COMPONENTS=15维)并提取128维特征
    nn.Linear(ART_COMPONENTS, 128),  # 全连接层，输入维度15，输出维度128
    # 输入：可能来自真实艺术家或生成器G的艺术作品
    nn.ReLU(),  # ReLU激活函数
    # 第二层：将128维特征映射到1维判别结果
    nn.Linear(128, 1),  # 全连接层，输入128，输出1
    nn.Sigmoid(),  # Sigmoid激活函数，将输出压缩到(0,1)区间
    # 1 = 确定是真实作品
    # 0 = 确定是生成作品
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()  # 开启交互模式，支持动态绘图
# 训练循环：10000次迭代
for step in range(10000):
    # 1. 准备真实数据
    artist_paintings = artist_works()  # 从真实艺术家获取一批艺术作品
    # 2. 准备生成器输入
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # 生成随机噪声向量(潜在空间采样) 随机灵感
    # torch.randn: 从标准正态分布采样
    # 3. 生成伪造艺术作品
    G_paintings = G(G_ideas)  # 生成器根据噪声产生伪造艺术作品
    # 4. 判别器评估
    prob_artist0 = D(artist_paintings)  # 判别器对真实艺术作品的评估
    # D试图最大化这个概率(接近1)
    prob_artist1 = D(G_paintings.detach())  # 判别器对伪造艺术作品的评估
    # D试图最小化这个概率(接近0)
    # 5. 计算判别器损失
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1 - prob_artist1))
    #更新判别器
    opt_D.zero_grad()
    D_loss.backward()  # retain_graph 这个参数是为了再次使用计算图纸
    opt_D.step()

    # 6. 训练生成器
    # 7. 计算生成器损失
    prob_artist1 = D(G_paintings)
    G_loss = torch.mean(torch.log(1. - prob_artist1)) # G试图让D对伪造作品输出高概率

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()  # 清除当前图像
        # 绘制生成器创作的作品(绿色)
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting')
        # 绘制真实作品的上边界(a=2时: y=2x²+1) (蓝色)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='Upper bound (a=2)')
        # 绘制真实作品的下边界(a=1时: y=x²) (红色)
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='Lower bound (a=1)')
        # 添加文本标注
        plt.text(-0.5, 2.3, f'D accuracy={prob_artist0.data.numpy().mean():.2f} (0.5 for D to converge)',
                 fontdict={'size': 15})
        plt.text(-0.5, 2.0, f'D score={-D_loss.data.numpy():.2f} (-1.38 for G to converge)',
                 fontdict={'size': 15})
        # 图像设置
        plt.ylim(0, 3)  # y轴范围
        plt.legend(loc='upper right', fontsize=12)  # 图例位置
        plt.draw()  # 更新绘制
        plt.pause(0.01)  # 暂停0.01秒，显示当前图像
plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终图像