import torch
import torch.utils.data as Data

def main():
    # 设置随机种子，保证每次运行结果可复现
    torch.manual_seed(1)
    # 定义批训练的数据个数
    BATCH_SIZE = 5  # 每个小批量包含5个样本
    # 创建输入数据x (1到10的10个等间距数)
    x = torch.linspace(1, 10, 10)
    # 创建目标数据y (10到1的10个等间距数，与x相反)
    y = torch.linspace(10, 1, 10)
    # 将数据转换为PyTorch的Dataset格式
    # TensorDataset用于包装数据和目标张量
    torch_dataset = Data.TensorDataset(x, y)

    # 创建DataLoader实现小批量训练
    loader = Data.DataLoader(
        dataset=torch_dataset,  # 使用的数据集
        batch_size=BATCH_SIZE,  # 每个小批量的大小
        shuffle=True,  # 是否打乱数据顺序
        num_workers=2,  # 使用2个子进程加载数据
    )

    # 训练循环：整个数据集训练3次(3个epoch)
    for epoch in range(3):
        # 遍历数据加载器，每次获取一个小批量
        # enumerate(loader)返回(step, (batch_x, batch_y))
        # step: 当前批次的序号
        # batch_x: 当前批次的输入数据
        # batch_y: 当前批次的目标数据
        for step, (batch_x, batch_y) in enumerate(loader):
            # 打印训练信息
            print('Epoch:', epoch,  # 当前训练轮次
                  '| Step:', step,  # 当前批次序号
                  '| batch x:', batch_x.numpy(),  # 当前批次输入数据
                  '| batch y:', batch_y.numpy())  # 当前批次目标数据


# Python多进程的标准写法：确保主模块被保护
if __name__ == '__main__':
    main()  # 执行主函数