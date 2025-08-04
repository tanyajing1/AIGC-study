import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import os
import random
import unicodedata  # 用于Unicode字符标准化
from tqdm import tqdm  # 用于显示进度条
import matplotlib.pyplot as plt  # 用于绘制损失曲线
from sklearn.model_selection import train_test_split  # 用于分割训练集和验证集


# --- 从cmn.txt格式文件加载数据 ---
def load_data_from_txt(file_path, max_lines=None):
    """从cmn.txt文件加载英文和中文句子对（格式：英文[TAB]中文[TAB]版权信息）。

    参数:
        file_path: 数据文件路径
        max_lines: 最大加载行数（None表示加载全部）

    返回:
        en_sentences: 英文句子列表
        zh_sentences: 中文句子列表
    """
    en_sentences, zh_sentences = [], []
    print(f"从{file_path}加载数据..." + (f"（最多{max_lines}行）" if max_lines else ""))

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：未在{file_path}找到数据文件")
        return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines_processed = 0  # 已处理的行数
            # 使用tqdm显示读取进度
            for line in tqdm(f, desc=f"读取{os.path.basename(file_path)}", total=max_lines if max_lines else None):
                # 达到最大行数限制则停止
                if max_lines is not None and lines_processed >= max_lines:
                    print(f"\n已达到{file_path}的最大行数限制{max_lines}。")
                    break

                # 分割英文、中文和版权信息（按制表符分割）
                parts = line.strip().split('\t')
                if len(parts) >= 2:  # 至少包含英文和中文部分
                    en = parts[0].strip()  # 英文句子
                    zh = parts[1].strip()  # 中文句子
                    if en and zh:  # 确保句子非空
                        en_sentences.append(en)
                        zh_sentences.append(zh)
                        lines_processed += 1
    except Exception as e:
        print(f"读取{file_path}时发生错误：{e}")
        return [], []

    print(f"从{file_path}加载了{len(en_sentences)}个句子对。")
    return en_sentences, zh_sentences


# --- 带字符标准化的词汇表类 ---
class Vocab:
    def __init__(self, sentences, min_freq=1, special_tokens=None):
        """构建词汇表，将字符映射到索引（支持特殊token和字符频率过滤）。

        参数:
            sentences: 用于构建词汇表的句子列表
            min_freq: 字符的最小出现频率（低于此频率的字符将被忽略）
            special_tokens: 特殊token列表（如<pad>、<unk>等）
        """
        self.stoi = {}  # 字符到索引的映射（string-to-index）
        self.itos = {}  # 索引到字符的映射（index-to-string）
        if special_tokens is None:
            # 默认特殊token：填充符、未知符、句首符、句尾符
            special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        self.special_tokens = special_tokens

        # 先初始化特殊token（确保它们在词汇表的最前面）
        idx = 0
        for token in special_tokens:
            self.stoi[token] = idx
            self.itos[idx] = token
            idx += 1

        # 统计字符频率（包含Unicode标准化）
        counter = {}  # 字符频率计数器
        print("统计字符频率以构建词汇表...")
        for s in tqdm(sentences, desc="处理句子构建词汇表"):
            if isinstance(s, str):
                # 对字符进行Unicode标准化（统一不同编码形式的相同字符）
                s = unicodedata.normalize('NFKC', s)
                for char in s:
                    # 英文字符转为小写（大小写归一化），其他字符保持原样
                    char = char.lower() if char.isalpha() else char
                    counter[char] = counter.get(char, 0) + 1  # 更新频率

        # 添加满足最小频率的字符到词汇表
        # 过滤掉已在特殊token中的字符
        non_special_counts = {token: count for token, count in counter.items() if token not in self.special_tokens}
        # 按频率从高到低排序
        sorted_tokens = sorted(non_special_counts.items(), key=lambda item: item[1], reverse=True)

        for token, count in tqdm(sorted_tokens, desc="构建词汇表映射"):
            # 只添加频率达标且未在词汇表中的字符
            if count >= min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

        # 确保<unk>（未知符）的映射正确（防止被其他字符覆盖）
        if '<unk>' in self.special_tokens:
            unk_intended_idx = self.special_tokens.index('<unk>')  # <unk>应在的位置
            # 检查当前映射是否正确
            if self.stoi.get('<unk>') != unk_intended_idx or self.itos.get(unk_intended_idx) != '<unk>':
                print(f"警告：<unk>的映射可能不一致，强制设置索引为{unk_intended_idx}。")
                # 移除现有<unk>的映射
                current_unk_mapping_val = self.stoi.pop('<unk>', None)
                if current_unk_mapping_val is not None and self.itos.get(current_unk_mapping_val) == '<unk>':
                    # 如果目标位置已有其他字符，先移除该字符的映射
                    if self.itos.get(unk_intended_idx) is not None and self.itos.get(unk_intended_idx) != '<unk>':
                        token_at_unk_idx = self.itos.get(unk_intended_idx)
                        if token_at_unk_idx in self.stoi and self.stoi[token_at_unk_idx] == unk_intended_idx:
                            del self.stoi[token_at_unk_idx]
                # 强制设置<unk>的映射
                self.stoi['<unk>'] = unk_intended_idx
                self.itos[unk_intended_idx] = '<unk>'

    def __len__(self):
        """返回词汇表大小"""
        return len(self.itos)


# --- 数据增强函数 ---
def augment_sentence(sentence, p=0.3):
    """简单的数据增强：随机删除字符或交换相邻字符。

    参数:
        sentence: 原始句子
        p: 执行增强的概率

    返回:
        增强后的句子（或原始句子，取决于概率）
    """
    if random.random() > p:  # 以1-p的概率不增强
        return sentence

    chars = list(sentence)  # 转为字符列表
    # 随机删除一个字符（句子长度>3时才执行）
    if random.random() < 0.3 and len(chars) > 3:
        del_idx = random.randint(0, len(chars) - 1)
        del chars[del_idx]

    # 随机交换相邻字符（句子长度>3时才执行）
    if random.random() < 0.3 and len(chars) > 3:
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    return ''.join(chars)  # 转回字符串


# --- 带数据增强的翻译数据集类 ---
class TranslationDataset(Dataset):
    def __init__(self, en_sentences, zh_sentences, src_vocab, tgt_vocab, augment=False):
        """翻译数据集，将中英文句子转换为索引序列，并支持数据增强。

        参数:
            en_sentences: 英文句子列表
            zh_sentences: 中文句子列表
            src_vocab: 源语言（英文）词汇表
            tgt_vocab: 目标语言（中文）词汇表
            augment: 是否启用数据增强
        """
        self.src_data = []  # 源语言句子的索引序列
        self.tgt_data = []  # 目标语言句子的索引序列
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.augment = augment  # 是否启用增强

        print("创建数据集张量...")
        # 提前获取特殊token的索引（避免重复查询）
        src_sos_idx = src_vocab.stoi['<sos>']  # 源语言句首符索引
        src_eos_idx = src_vocab.stoi['<eos>']  # 源语言句尾符索引
        src_unk_idx = src_vocab.stoi['<unk>']  # 源语言未知符索引
        tgt_sos_idx = tgt_vocab.stoi['<sos>']  # 目标语言句首符索引
        tgt_eos_idx = tgt_vocab.stoi['<eos>']  # 目标语言句尾符索引
        tgt_unk_idx = tgt_vocab.stoi['<unk>']  # 目标语言未知符索引

        # 遍历句子对，转换为索引序列（带进度条）
        for en, zh in tqdm(zip(en_sentences, zh_sentences), total=len(en_sentences), desc="向量化数据"):
            # 源语言序列：<sos> + 字符索引 + <eos>
            src_ids = [src_sos_idx] + [src_vocab.stoi.get(c, src_unk_idx) for c in en] + [src_eos_idx]
            # 目标语言序列：<sos> + 字符索引 + <eos>
            tgt_ids = [tgt_sos_idx] + [tgt_vocab.stoi.get(c, tgt_unk_idx) for c in zh] + [tgt_eos_idx]
            # 转换为张量并存储
            self.src_data.append(torch.LongTensor(src_ids))
            self.tgt_data.append(torch.LongTensor(tgt_ids))
        print("数据集张量创建完成。")

    def __len__(self):
        """返回数据集大小"""
        return len(self.src_data)

    def __getitem__(self, idx):
        """获取索引对应的样本（支持数据增强）"""
        src, tgt = self.src_data[idx], self.tgt_data[idx]
        # 以50%概率对源语言句子应用增强（仅当启用增强时）
        if self.augment and random.random() < 0.5:
            # 将索引序列转回字符串
            src_str = ''.join([self.src_vocab.itos[i] for i in src.tolist()])
            # 增强字符串
            src_str = augment_sentence(src_str)
            # 重新转换为索引序列
            src = torch.LongTensor([self.src_vocab.stoi.get(c, self.src_vocab.stoi['<unk>']) for c in src_str])
        return src, tgt


# --- 批次数据处理函数 ---
def collate_fn(batch, pad_idx=0):
    """对批次内的序列进行填充，使长度一致（用于DataLoader）。

    参数:
        batch: 一批数据（列表，每个元素为(src, tgt)元组）
        pad_idx: 填充符的索引

    返回:
        填充后的源语言批次和目标语言批次（均为张量）
    """
    src_batch, tgt_batch = zip(*batch)  # 分离源语言和目标语言数据
    # 使用pad_sequence填充序列，使同一批次长度一致（batch_first=True表示第一维为批次）
    src_batch_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch_padded = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)
    return src_batch_padded, tgt_batch_padded


# --- 掩码创建函数 ---
def create_masks(src, tgt, pad_idx):
    """为源序列和目标序列创建掩码（用于Transformer，避免注意力关注无效位置）。

    参数:
        src: 源语言序列张量 (batch_size, src_seq_len)
        tgt: 目标语言序列张量 (batch_size, tgt_seq_len)
        pad_idx: 填充符的索引

    返回:
        src_mask: 源序列掩码（过滤填充符）
        tgt_mask: 目标序列掩码（过滤填充符+防止关注未来token）
    """
    device = src.device  # 获取设备（CPU/GPU）

    # 源序列填充掩码：(批次大小, 1, 1, 源序列长度)
    # 填充位置为False，有效位置为True
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # 目标序列掩码
    # 目标填充掩码：(批次大小, 1, 目标序列长度, 1)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(-1)

    # 前瞻掩码（下三角矩阵）：防止模型关注未来的token
    tgt_seq_length = tgt.size(1)  # 目标序列长度
    # 生成上三角矩阵（对角线以上为1），取反后得到下三角掩码（有效位置为True）
    look_ahead_mask = (1 - torch.triu(
        torch.ones((tgt_seq_length, tgt_seq_length), device=device), diagonal=1
    )).bool().unsqueeze(0).unsqueeze(0)  # 扩展维度适配批次

    # 目标掩码：填充掩码和前瞻掩码的交集（同时过滤填充和未来token）
    tgt_mask = tgt_pad_mask & look_ahead_mask

    return src_mask.to(device), tgt_mask.to(device)


# --- 标签平滑损失 ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.1):
        """标签平滑损失（防止模型对预测过度自信）。

        参数:
            classes: 类别总数（词汇表大小）
            padding_idx: 填充符的索引（忽略该位置的损失）
            smoothing: 平滑系数（0表示不平滑，即交叉熵损失）
        """
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')  # KL散度损失（批次均值）
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # 真实标签的置信度
        self.smoothing = smoothing  # 平滑到其他标签的概率
        self.classes = classes  # 类别数
        self.log_softmax = nn.LogSoftmax(dim=1)  # 计算log(softmax(x))

    def forward(self, x, target):
        """计算损失。

        参数:
            x: 模型输出 (batch_size*seq_len, classes)
            target: 真实标签 (batch_size*seq_len,)
        """
        assert x.size(1) == self.classes  # 确保输出维度与类别数一致
        x = self.log_softmax(x)  # 先计算log_softmax

        # 构建平滑后的目标分布
        true_dist = x.data.clone()  # 初始化分布
        # 每个非真实标签分配smoothing/(classes-2)的概率（减2是排除真实标签和填充符）
        true_dist.fill_(self.smoothing / (self.classes - 2))
        # 真实标签位置分配confidence的概率
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 填充符位置的概率设为0（忽略填充）
        true_dist[:, self.padding_idx] = 0
        # 找到填充符的位置，将其分布清零
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # 计算KL散度损失（等价于带平滑的交叉熵）
        return self.criterion(x, true_dist)


# --- 带预热的学习率调度器 ---
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        """带预热的学习率调度器（Transformer原论文中的调度方式）。

        参数:
            optimizer: 优化器
            d_model: 模型维度（用于计算学习率）
            warmup_steps: 预热步数（前warmup_steps学习率递增，之后递减）
        """
        self.optimizer = optimizer
        self.d_model = d_model  # 模型维度
        self.warmup_steps = warmup_steps  # 预热步数
        self.current_step = 0  # 当前训练步数

    def step(self):
        """更新学习率（每步调用一次）"""
        self.current_step += 1
        # 学习率计算公式：(d_model^-0.5) * min(step^-0.5, step*warmup_steps^-1.5)
        lr = (self.d_model ** -0.5) * min(
            self.current_step ** -0.5,
            self.current_step * self.warmup_steps ** -1.5
        )
        # 应用学习率到优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# --- 可视化函数 ---
def plot_losses(train_losses, val_losses=None, save_path='training_loss.png'):
    """绘制训练和验证损失曲线并保存。

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        save_path: 图像保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', marker='o')  # 绘制训练损失
    if val_losses is not None:
        plt.plot(val_losses, label='验证损失', marker='o')  # 绘制验证损失
    plt.title('训练和验证损失曲线')
    plt.xlabel('轮次（Epoch）')
    plt.ylabel('损失值')
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图像（释放资源）
    print(f"损失曲线已保存到{save_path}")


# --- 主执行模块 ---
if __name__ == '__main__':
    # --- 配置参数 ---
    DATA_PATH = 'cmn.txt'  # 数据文件路径
    MODEL_SAVE_PATH = 'best_model_cmn.pth'  # 最佳模型保存路径

    # 数据限制
    MAX_TOTAL_LINES = None  # 最大加载行数（None表示全部）
    VALID_RATIO = 0.2  # 验证集比例

    # 超参数
    BATCH_SIZE = 16  # 批次大小
    NUM_EPOCHS = 15  # 训练轮次
    LEARNING_RATE = 5e-5  # 初始学习率（实际由调度器调整）
    D_MODEL = 128  # 模型维度（Transformer中各层的隐藏维度）
    NUM_HEADS = 4  # 注意力头数
    NUM_LAYERS = 4  # Transformer编码器/解码器的层数
    D_FF = 512  # 前馈网络的隐藏层维度
    DROPOUT = 0.3  # dropout概率
    MIN_FREQ = 2  # 词汇表字符的最小出现频率
    PRINT_FREQ = 100  # 打印频率（未使用，此处保留）
    WARMUP_STEPS = 2000  # 学习率预热步数
    GRAD_ACCUM_STEPS = 8  # 梯度累积步数（模拟更大批次）
    USE_LABEL_SMOOTHING = True  # 是否使用标签平滑损失

    # 设备配置（优先使用GPU）
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{DEVICE}")

    # --- 加载并分割数据 ---
    print(f"从{DATA_PATH}加载数据...")
    en_sentences, zh_sentences = load_data_from_txt(DATA_PATH, max_lines=MAX_TOTAL_LINES)

    if not en_sentences:
        print("未加载到数据，程序退出。")
        exit()

    # 分割训练集和验证集
    train_en, val_en, train_zh, val_zh = train_test_split(
        en_sentences, zh_sentences,
        test_size=VALID_RATIO,  # 验证集占比
        random_state=42  # 随机种子（保证结果可复现）
    )

    print(f"训练集大小：{len(train_en)}，验证集大小：{len(val_en)}")

    # --- 构建词汇表 ---
    print("从训练数据构建词汇表...")
    src_vocab = Vocab(train_en, min_freq=MIN_FREQ)  # 源语言（英文）词汇表
    tgt_vocab = Vocab(train_zh, min_freq=MIN_FREQ)  # 目标语言（中文）词汇表
    print(f"源语言词汇表大小：{len(src_vocab)}")
    print(f"目标语言词汇表大小：{len(tgt_vocab)}")

    # 检查填充符索引是否为0（后续代码依赖此设定）
    PAD_IDX = src_vocab.stoi['<pad>']
    if PAD_IDX != 0 or tgt_vocab.stoi['<pad>'] != 0:
        print("错误：填充符索引不为0，需调整collate_fn和损失函数。")
        exit()

    # --- 创建数据集 ---
    print("创建训练数据集...")
    train_dataset = TranslationDataset(train_en, train_zh, src_vocab, tgt_vocab, augment=True)  # 启用增强
    print("创建验证数据集...")
    val_dataset = TranslationDataset(val_en, val_zh, src_vocab, tgt_vocab)  # 不启用增强

    # 创建数据加载器（自动批处理和打乱）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练集打乱
        collate_fn=lambda b: collate_fn(b, PAD_IDX)  # 使用自定义填充函数
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )

    # --- 初始化模型 ---
    print("初始化模型...")
    from transformer import Transformer  # 导入Transformer模型（假设在model.py中定义）

    model = Transformer(
        src_vocab_size=len(src_vocab),  # 源语言词汇表大小
        tgt_vocab_size=len(tgt_vocab),  # 目标语言词汇表大小
        d_model=D_MODEL,  # 模型维度
        num_heads=NUM_HEADS,  # 注意力头数
        num_layers=NUM_LAYERS,  # 编码器/解码器层数
        d_ff=D_FF,  # 前馈网络维度
        dropout=DROPOUT  # dropout概率
    ).to(DEVICE)  # 移动到指定设备


    def count_parameters(model):
        """计算模型可训练参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'模型可训练参数数量：{count_parameters(model):,}')

    # --- 初始化优化器和损失函数 ---
    # Adam优化器（Transformer原论文推荐参数）
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    # 带预热的学习率调度器
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)

    # 选择损失函数
    if USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 普通交叉熵（忽略填充符）

    # --- 训练循环 ---
    best_val_loss = float('inf')  # 记录最佳验证损失（初始为无穷大）
    print("开始训练...")

    train_losses = []  # 记录每轮训练损失
    val_losses = []  # 记录每轮验证损失

    for epoch in range(NUM_EPOCHS):
        # 训练模式
        model.train()
        epoch_loss = 0  # 累计本轮训练损失
        optimizer.zero_grad()  # 清零梯度
        # 训练进度条
        train_iterator = tqdm(train_loader, desc=f"第{epoch + 1}/{NUM_EPOCHS}轮训练")

        for i, (src, tgt) in enumerate(train_iterator):
            # 移动数据到设备
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            # 目标输入（不含最后一个字符）和目标输出（不含第一个字符）
            tgt_input = tgt[:, :-1]  # 用于输入模型
            tgt_output = tgt[:, 1:]  # 用于计算损失（真实标签）

            # 创建掩码
            src_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX)
            # 模型前向传播（得到logits）
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # 调整形状以适配损失函数
            output_dim = logits.shape[-1]  # 输出维度（目标词汇表大小）
            logits_reshaped = logits.contiguous().view(-1, output_dim)  # 展平为(batch*seq_len, vocab_size)
            tgt_output_reshaped = tgt_output.contiguous().view(-1)  # 展平为(batch*seq_len,)

            # 计算损失（梯度累积：损失除以累积步数）
            loss = criterion(logits_reshaped, tgt_output_reshaped) / GRAD_ACCUM_STEPS
            loss.backward()  # 反向传播计算梯度

            # 梯度累积：每GRAD_ACCUM_STEPS步更新一次参数
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪（防止梯度爆炸）
                optimizer.step()  # 更新参数
                scheduler.step()  # 更新学习率
                optimizer.zero_grad()  # 清零梯度

            # 累计损失（乘以累积步数还原真实损失）
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            # 在进度条显示当前损失
            train_iterator.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)

        # 计算本轮平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证模式
        model.eval()
        val_loss = 0  # 累计本轮验证损失
        val_iterator = tqdm(val_loader, desc=f"第{epoch + 1}/{NUM_EPOCHS}轮验证")
        with torch.no_grad():  # 关闭梯度计算（节省内存和时间）
            for src, tgt in val_iterator:
                # 移动数据到设备
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                # 目标输入和输出
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                # 创建掩码
                src_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX)
                # 模型前向传播
                logits = model(src, tgt_input, src_mask, tgt_mask)
                # 调整形状
                output_dim = logits.shape[-1]
                logits_reshaped = logits.contiguous().view(-1, output_dim)
                tgt_output_reshaped = tgt_output.contiguous().view(-1)
                # 计算损失（不累积梯度）
                loss = criterion(logits_reshaped, tgt_output_reshaped)
                val_loss += loss.item()
                # 在进度条显示当前损失
                val_iterator.set_postfix(loss=loss.item())

        # 计算本轮平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # 打印本轮损失 summary
        print(f'\n第{epoch + 1}轮总结：训练损失：{avg_train_loss:.4f}，验证损失：{avg_val_loss:.4f}')

        # 保存最佳模型（验证损失最低时）
        if avg_val_loss < best_val_loss:
            print(
                f"验证损失下降（{best_val_loss:.4f} --> {avg_val_loss:.4f}）。保存模型到{MODEL_SAVE_PATH}...")
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),  # 模型参数
                'src_vocab': src_vocab,  # 源语言词汇表
                'tgt_vocab': tgt_vocab,  # 目标语言词汇表
                'epoch': epoch,  # 当前轮次
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
                'loss': best_val_loss,  # 最佳损失
                'config': {  # 模型配置（用于后续加载）
                    'd_model': D_MODEL, 'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS,
                    'd_ff': D_FF, 'dropout': DROPOUT,
                    'src_vocab_size': len(src_vocab), 'tgt_vocab_size': len(tgt_vocab)
                }
            }, MODEL_SAVE_PATH)

    # --- 生成可视化结果 ---
    print("\n生成训练可视化结果...")
    plot_losses(train_losses, val_losses, save_path='training_validation_loss1.png')

    print("训练完成！")