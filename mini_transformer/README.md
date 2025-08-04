这是一个基于 PyTorch 实现的轻量级 Transformer 模型，主要用于中英文翻译任务。
项目包含完整的模型定义、训练流程和预测功能。

# 本项目实现了经典的 Transformer 模型（基于《Attention Is All You Need》论文），包含以下核心功能：
    完整的 Transformer 编码器 - 解码器架构
    位置编码（Positional Encoding）实现
    多头注意力机制（Multi-Head Attention）
    数据预处理与词汇表构建
    训练脚本与损失可视化
    模型预测与翻译示例

# 环境依赖
    Python 3.7+
    PyTorch 1.8+
    numpy
    tqdm
    matplotlib
    scikit-learn

# 数据准备
    项目使用类似cmn.txt格式的平行语料库（英文 \t 中文 \t 注释），数据格式示例：
    plaintext
    I love you.	我爱你。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CMN) & #1253818 (eng)
    Hello world.	你好，世界。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CMN) & #1253818 (eng)

# 模型训练
    修改train.py中的训练参数（可选）：
        批次大小（batch size）
        训练轮次（epochs）
        模型超参数（d_model, num_heads 等）
        数据增强开关
    
# 运行训练脚本：
    python train.py
    训练过程中会：
        自动构建中英文词汇表
        保存最佳模型到best_model_cmn.pth
        生成损失曲线图片training_loss.png

# 翻译预测
    确保训练好的模型best_model_cmn.pth在项目根目录
    运行预测脚本：
    python predict.py
        脚本会自动加载模型并对测试句子进行翻译，示例输出：
        plaintext
        --- Starting Translation Examples ---
        --------------------
        Input:      I love you.
        Translation: 我爱你。
        --------------------
        Input:      Wait
        Translation: 等等
        ...
        你也可以修改predict.py中的test_sentences列表，添加自己想要翻译的句子。

# 项目结构
    plaintext
    mini_transformer/
    ├── model.py        # Transformer模型定义
    │   ├── PositionalEncoding  # 位置编码模块
    │   ├── MultiHeadAttention  # 多头注意力模块
    │   ├── EncoderLayer/DecoderLayer  # 编码/解码层
    │   └── Transformer  # 完整模型组装
    ├── train.py        # 训练相关代码
    │   ├── 数据加载与预处理
    │   ├── Vocab 词汇表类
    │   ├── 数据集与数据增强
    │   ├── 损失函数（含标签平滑）
    │   └── 训练循环与学习率调度
    └── predict.py      # 预测与翻译功能
        ├── 模型加载
        ├── 翻译函数
        └── 示例测试

# 模型特点
    实现了原始 Transformer 的核心结构
    使用标签平滑（Label Smoothing）提高泛化能力
    采用学习率预热（Warmup）策略
    支持数据增强（随机删除、字符交换）
    完整的掩码机制（Padding Mask + Look-ahead Mask）

# ⚠️问题
    1、选择数据集的时候，一开始选择了大数据集，但是训练却只使用了其中一小部分，导致了后来预测的时候会一直输出重复、无意义的短语。这说明模型训练还不够充分。
    但是用大量数据cpu计算的时间非常久，后面选择用小数据集，可是小数据集的单词有限，最终训练效果也一般
    2、一开始认为训练效果一般是由于我分词没有处理好，后面修改了载入数据集的分词处理情况，使得英文按空格拆分，并单独保留标点，中文使用工具jieba进行词级分词。
    但是效果并不好，在预测翻译的时候产生了大量的 <unk> 标记，这表明模型无法正确识别输入单词或生成正确的输出字符。
    3、在其中一版模型训练的时候，loss的值已经降到0.7了，可是最后的预测效果也不是很理想
    4、高频词（如"hello"）仅有6条样本，导致模型无法学习多场景翻译。常见词和句式覆盖不足，而其他低频词可能干扰模型对基础词汇的学习。

# 参考资料
    Attention Is All You Need
    PyTorch 官方文档
    b站【一小时从函数到Transformer！一路大白话彻底理解AI原理-哔哩哔哩】 https://b23.tv/bdyGhDM
    transformer逐层分解：https://v11enp9ok1h.feishu.cn/wiki/RbIpw8NJHijTJXkQAqncTmRunBQ?from=from_copylink
    b站【transform代码pytorch实现机器翻译-哔哩哔哩】 https://b23.tv/Y0eX2uq
    CSDN https://blog.csdn.net/qq_20144897/article/details/138419007?fromshare=blogdetail&sharetype=blogdetail&sharerId=138419007&sharerefer=PC&sharesource=2201_75767444&sharefrom=from_link
    数据集来源：飞浆https://aistudio.baidu.com/datasetdetail/78721