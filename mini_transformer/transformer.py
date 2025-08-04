import torch
import torch.nn as nn
import math

#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵 (max_len × d_model)
        pe = torch.zeros(max_len, d_model)

        # 创建位置序列 (0到max_len-1)，并转换为列向量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于生成正弦和余弦函数的频率
        # 使用指数函数和对数函数来实现公式中的分母部分
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 对偶数位置使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数位置使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一个维度，便于批量处理 (1 × max_len × d_model)
        pe = pe.unsqueeze(0)
        # 位置编码固定，不更新参数
        # 保存模型时会保存缓冲区，在引入模型时缓冲区也被引入
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 不计算梯度
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x

#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0 # 确保d_model可以被num_heads整除
        self.d_k = d_model // num_heads # 每个头的维度
        self.num_heads = num_heads
        # 定义Q、K、V和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model) # 查询变换
        self.W_k = nn.Linear(d_model, d_model) # 键变换
        self.W_v = nn.Linear(d_model, d_model) # 值变换
        self.W_o = nn.Linear(d_model, d_model) # 输出变换

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 线性变换并分割为多个头
        # 1. 通过线性层投影
        # 2. 重塑为 (batch_size, seq_len, num_heads, d_k)
        # 3. 转置为 (batch_size, num_heads, seq_len, d_k) 以便计算注意力
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 计算注意力分数 (QK^T)/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 计算注意力权重 (softmax)
        attn_weights = torch.softmax(scores, dim=-1)
        # 计算上下文向量 (注意力权重与V相乘)
        context = torch.matmul(attn_weights, V)
        # 合并多头输出
        # 1. 转置回 (batch_size, seq_len, num_heads, d_k)
        # 2. 连续化内存
        # 3. 重塑为 (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        return self.W_o(context)

#编码器
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力子层
        self.attn = MultiHeadAttention(d_model, num_heads)
        # 前馈神经网络子层
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), # 扩展维度
            nn.ReLU(), # 激活函数
            nn.Linear(d_ff, d_model) # 降回原维度
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 第一子层: 多头注意力 + Add & Norm
        attn_output = self.attn(x, x, x, mask) # 自注意力(Q=K=V)
        x = self.norm1(x + self.dropout(attn_output)) # 残差连接 + dropout + 层归一化
        # 第二子层: 前馈网络 + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

#解码器
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 自注意力子层
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 编码器-解码器注意力子层
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # 三个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # 前馈神经网络子层
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 第一子层: 自注意力 + Add & Norm (带目标mask)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 第二子层: 编码器-解码器注意力 + Add & Norm
        # Q来自解码器，K和V来自编码器
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # 第三子层: 前馈网络 + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        # 编码器和解码器的词嵌入层
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 编码器堆叠多层
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 解码器堆叠多层
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 输出线性层 (将解码器输出映射到目标词汇表大小)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        # 编码器和解码器的最终层归一化
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def encode(self, src, src_mask):
        # 源语言词嵌入
        src_embeded = self.encoder_embed(src)
        # 添加位置编码
        src = self.pos_encoder(src_embeded)
        # 通过多层编码器
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        # 最终层归一化
        return self.encoder_norm(src)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        # 目标语言词嵌入
        tgt_embeded = self.decoder_embed(tgt)
        # 添加位置编码
        tgt = self.pos_encoder(tgt_embeded)
        # 通过多层解码器
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        # 最终层归一化
        return self.decoder_norm(tgt)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        logits = self.fc_out(dec_output)
        return logits
