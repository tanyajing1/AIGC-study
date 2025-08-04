import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import torch.serialization
# 从训练脚本导入Vocab类（词汇表处理）
from train import Vocab

# 添加安全全局变量，确保Vocab类能被正确反序列化
torch.serialization.add_safe_globals([Vocab])

# --- 导入必要的模型组件 ---
try:
    # 导入Transformer模型、位置编码和其他工具函数
    from transformer import Transformer, PositionalEncoding
    from train import Vocab, create_masks
except ImportError as e:
    print(f"导入模块错误: {e}")
    print("请确保 transformer.py 和 train.py 在 Python 路径中且包含必要的定义")
    sys.exit(1)

# --- 配置参数 ---
CHECKPOINT_PATH = 'best_model_cmn.pth'  # 预训练模型检查点路径
MAX_LENGTH = 60  # 翻译结果的最大长度（防止生成过长序列）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备（优先GPU）

print(f"使用设备: {DEVICE}")
print(f"从路径加载检查点: {CHECKPOINT_PATH}")

# 检查模型文件是否存在
if not os.path.exists(CHECKPOINT_PATH):
    print(f"错误: 检查点文件未找到 {CHECKPOINT_PATH}")
    sys.exit(1)

# --- 加载模型检查点（包含模型参数和词汇表） ---
try:
    # 加载检查点，map_location确保模型加载到指定设备，weights_only=False允许加载非权重数据（如词汇表）
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print("检查点加载成功")
except Exception as e:
    print(f"加载检查点文件错误: {e}")
    sys.exit(1)

# --- 验证检查点内容是否完整 ---
# 检查点必须包含的关键信息：模型参数、源语言词汇表、目标语言词汇表、模型配置
required_keys = ['model_state_dict', 'src_vocab', 'tgt_vocab']
if 'config' in checkpoint:
    required_keys.append('config')  # 若存在配置信息则也需验证

for key in required_keys:
    if key not in checkpoint:
        print(f"错误: 检查点中缺少必要的键 '{key}'")
        sys.exit(1)

# --- 提取词汇表和模型配置 ---
try:
    # 从检查点中获取源语言（如英文）和目标语言（如中文）词汇表
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    # 验证词汇表类型是否正确
    assert isinstance(src_vocab, Vocab) and isinstance(tgt_vocab, Vocab)
    # 获取填充符<pad>的索引（用于掩码）
    PAD_IDX = src_vocab.stoi.get('<pad>', 0)

    # 从检查点加载模型配置（训练时的超参数）
    if 'config' in checkpoint:
        config = checkpoint['config']
        D_MODEL = config['d_model']  # 模型维度
        NUM_HEADS = config['num_heads']  # 注意力头数
        NUM_LAYERS = config['num_layers']  # 编码器/解码器层数
        D_FF = config['d_ff']  # 前馈网络维度
        DROPOUT = config['dropout']  # dropout概率
        SRC_VOCAB_SIZE = config['src_vocab_size']  # 源语言词汇表大小
        TGT_VOCAB_SIZE = config['tgt_vocab_size']  # 目标语言词汇表大小
        print("从检查点加载模型配置")

        # 验证配置中的词汇表大小与实际加载的是否一致
        if SRC_VOCAB_SIZE != len(src_vocab) or TGT_VOCAB_SIZE != len(tgt_vocab):
            print("警告: 配置中的词汇表大小与加载的词汇表不匹配!")
            # 以实际加载的词汇表大小为准
            SRC_VOCAB_SIZE = len(src_vocab)
            TGT_VOCAB_SIZE = len(tgt_vocab)
    else:
        # 若检查点中无配置，使用默认超参数（不推荐，可能与训练时不一致）
        print("警告: 检查点中未找到模型配置。使用手动定义参数")
        D_MODEL = 512
        NUM_HEADS = 8
        NUM_LAYERS = 6
        D_FF = 2048
        DROPOUT = 0.1
        SRC_VOCAB_SIZE = len(src_vocab)
        TGT_VOCAB_SIZE = len(tgt_vocab)

    print(f"源词汇表大小: {len(src_vocab)}")
    print(f"目标词汇表大小: {len(tgt_vocab)}")
except Exception as e:
    print(f"处理词汇表或配置时出错: {e}")
    sys.exit(1)

# --- 初始化Transformer模型 ---
try:
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,  # 源语言词汇表大小
        tgt_vocab_size=TGT_VOCAB_SIZE,  # 目标语言词汇表大小
        d_model=D_MODEL,  # 模型维度
        num_heads=NUM_HEADS,  # 注意力头数
        num_layers=NUM_LAYERS,  # 编码器/解码器层数
        d_ff=D_FF,  # 前馈网络维度
        dropout=DROPOUT  # dropout概率
    ).to(DEVICE)  # 将模型移动到指定设备
    print("模型初始化成功")


    # 计算模型参数数量的辅助函数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    print(f'模型共有 {count_parameters(model):,} 个参数')

except Exception as e:
    print(f"初始化Transformer模型错误: {e}")
    sys.exit(1)

# --- 加载模型训练好的参数 ---
try:
    # 将检查点中的模型参数加载到当前模型
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置模型为评估模式（关闭dropout等训练特有层）
    print("模型状态加载成功")
except RuntimeError as e:
    print(f"加载模型state_dict错误: {e}")
    print("这表明加载的检查点架构与当前初始化的模型不匹配")
    print("请验证超参数(D_MODEL, NUM_HEADS等)是否与训练时完全一致")
    sys.exit(1)
except Exception as e:
    print(f"加载模型状态时发生意外错误: {e}")
    sys.exit(1)


# --- 核心翻译函数 ---
def translate(sentence: str, model: nn.Module, src_vocab: Vocab, tgt_vocab: Vocab,
              device: torch.device, max_length: int = 50):
    """使用训练好的Transformer模型将源语言句子翻译为目标语言

    参数:
        sentence: 源语言输入句子（如英文）
        model: 训练好的Transformer模型
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        device: 运行设备（CPU/GPU）
        max_length: 生成翻译的最大长度（防止无限循环）

    返回:
        翻译后的目标语言句子（如中文）
    """
    model.eval()  # 确保模型在评估模式

    # 检查输入是否为有效字符串
    if not isinstance(sentence, str):
        return "[错误: 输入类型无效]"

    # 获取源语言特殊符号的索引
    src_sos_idx = src_vocab.stoi.get('<sos>')  # 句首符
    src_eos_idx = src_vocab.stoi.get('<eos>')  # 句尾符
    src_unk_idx = src_vocab.stoi.get('<unk>', 0)  # 未知符
    src_pad_idx = src_vocab.stoi.get('<pad>', 0)  # 填充符

    # 检查源语言词汇表是否包含必要符号
    if src_sos_idx is None or src_eos_idx is None:
        return "[错误: 源词汇表问题]"

    # 处理输入句子：添加句首/句尾符，转换为字符列表
    src_tokens = ['<sos>'] + list(sentence) + ['<eos>']
    # 将字符转换为索引（未知字符用<unk>）
    src_ids = [src_vocab.stoi.get(token, src_unk_idx) for token in src_tokens]
    # 转换为张量并添加批次维度（模型要求输入为[batch_size, seq_len]）
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)

    # 创建源语言掩码（过滤填充符，确保注意力不关注填充位置）
    src_mask = (src_tensor != src_pad_idx).unsqueeze(1).unsqueeze(2).to(device)

    # 编码源语言句子（不计算梯度，节省资源）
    with torch.no_grad():
        try:
            enc_output = model.encode(src_tensor, src_mask)  # 编码器输出
        except Exception as e:
            print(f"模型编码错误: {e}")
            return "[错误: 编码失败]"

    # 获取目标语言特殊符号的索引
    tgt_sos_idx = tgt_vocab.stoi.get('<sos>')  # 句首符
    tgt_eos_idx = tgt_vocab.stoi.get('<eos>')  # 句尾符
    tgt_pad_idx = tgt_vocab.stoi.get('<pad>', 0)  # 填充符

    # 检查目标语言词汇表是否包含必要符号
    if tgt_sos_idx is None or tgt_eos_idx is None:
        return "[错误: 目标词汇表问题]"

    # 初始化目标序列：从句首符开始
    tgt_ids = [tgt_sos_idx]

    # 循环生成目标语言序列（最多max_length步）
    for i in range(max_length):
        # 将当前目标序列转换为张量并添加批次维度
        tgt_tensor = torch.LongTensor(tgt_ids).unsqueeze(0).to(device)
        tgt_len = tgt_tensor.size(1)  # 当前目标序列长度

        # 创建目标语言掩码：
        # 1. 填充掩码：过滤填充符
        tgt_pad_mask = (tgt_tensor != tgt_pad_idx).unsqueeze(1).unsqueeze(-1)
        # 2. 前瞻掩码：防止模型关注未来的字符（确保自回归生成）
        look_ahead_mask = (1 - torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1  # 上三角矩阵（对角线以上为1）
        )).bool().unsqueeze(0).unsqueeze(0)  # 扩展维度适配批次
        # 合并掩码：同时过滤填充符和未来字符
        combined_tgt_mask = tgt_pad_mask & look_ahead_mask

        # 解码过程（不计算梯度）
        with torch.no_grad():
            try:
                # 解码器输出：使用编码器结果和当前目标序列
                output = model.decode(tgt_tensor, enc_output, src_mask, combined_tgt_mask)
                # 取最后一个字符的输出，通过全连接层得到词汇表分布
                logits = model.fc_out(output[:, -1, :])
            except Exception as e:
                print(f"解码步骤 {i} 错误: {e}")
                return "[错误: 解码失败]"

        # 选择概率最高的字符作为下一个预测
        pred_token_id = logits.argmax(1).item()
        tgt_ids.append(pred_token_id)

        # 若预测到句尾符，则停止生成
        if pred_token_id == tgt_eos_idx:
            break

    # 过滤特殊符号（<sos>、<eos>、<pad>）
    special_indices = {
        tgt_vocab.stoi.get(tok, -999)
        for tok in ['<sos>', '<eos>', '<pad>']
    }
    # 将索引转换为字符（忽略特殊符号）
    translated_tokens = [tgt_vocab.itos.get(idx, '<unk>') for idx in tgt_ids if idx not in special_indices]

    # 拼接字符得到最终翻译结果
    return "".join(translated_tokens)


# 测试句子列表（英文）
test_sentences = [
    "I love you.",
    "Wait",
    "He like dog.",
    "I like to work",
    "Try it.",
    "I'm pretty busy.",
]

print("\n--- 开始翻译示例 ---")
for sentence in test_sentences:
    print("-" * 20)
    print(f"输入:      {sentence}")
    # 调用翻译函数
    translation = translate(sentence, model, src_vocab, tgt_vocab, DEVICE, max_length=MAX_LENGTH)
    print(f"翻译结果: {translation}")

print("-" * 20)
print("翻译完成。")