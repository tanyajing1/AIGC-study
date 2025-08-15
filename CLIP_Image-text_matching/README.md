# 🚀图文匹配搜索系统
    基于 CLIP 模型和 Milvus 向量数据库构建的图文跨模态匹配搜索系统，支持通过文本描述或图像示例搜索相似图像。
# ✨项目概述
    本项目利用 OpenAI 的 CLIP 模型实现跨模态特征提取，将文本和图像转换为统一维度的向量空间，并使用Milvus向量数据库进行高效的相似性搜索，最终通过Gradio提供直观的可视化交互界面。
# ✨核心功能
    文本到图像搜索：通过输入文本描述，查找最匹配的图像
    图像到图像搜索：通过上传参考图像，查找视觉相似的图像
    批量图像向量导入：支持将本地图像数据集批量转换为向量并存储到 Milvus
# ⭐️环境要求
    Python 3.8+
    Milvus 2.x（向量数据库服务）
    torch >= 1.10.0
    transformers >= 4.20.0
    pymilvus >= 2.2.0
    gradio >= 3.0.0
    pillow >= 9.0.0
    pyyaml >= 6.0.0# 图文匹配搜索系统
# ☀️安装步骤:
1. 安装Milvus服务
    参考官方文档安装 Milvus standalone 版：Milvus 安装指南
    操作指南参考csdn：https://blog.csdn.net/m0_37721946/article/details/147757267?fromshare=blogdetail&sharetype=blogdetail&sharerId=147757267&sharerefer=PC&sharesource=2201_75767444&sharefrom=from_link
2. 克隆项目并安装依赖
    # 安装依赖
       pip install torch transformers pymilvus gradio pillow pyyaml numpy
3. 配置文件设置
修改cfg/config.yaml文件：
    data_path: /path/to/your/image/dataset  # 本地图像数据集路径
    model: openai/clip-vit-base-patch32  # 可使用HuggingFace模型ID或本地模型路径
4. 数据集下载：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

# 🌟使用说明（运行方式）
1. 初始化 Milvus 数据库
    # 首次使用需创建数据库、集合和索引：
        bash
        python test_connect.py
注意：根据需要解除test_connect.py中主函数里对应步骤的注释

2. 批量导入图像数据
    # 将图像数据集转换为向量并导入 Milvus：
        bash
        python insert_search.py
程序会递归处理config.yaml中data_path下的所有子目录中的 JPG 图像

3. 启动搜索界面
# 
    bash
    python main.py
启动后访问 http://127.0.0.1:6006 即可使用搜索功能

# ✨项目结构
    CLIP_Image-text_matching
    ├── cfg/
    │   └── config.yaml         # 配置文件（数据路径、模型路径等）
    ├── scripts/
    │   └── insert_search.py    # 核心功能模块（模型初始化、向量处理、搜索逻辑）
    │   └── main.py                 # Gradio交互界面
    │   └──test_connect.py         # Milvus数据库初始化工具（创建库、集合、索引等）
    └── README.md               # 项目说明文档
# ⚡️核心模块说明
    insert_search.py：实现 CLIP 模型加载、文本 / 图像向量生成、Milvus 交互、批量插入和搜索功能
    main.py：基于 Gradio 构建的 Web 交互界面，支持文本输入和图像上传
    test_connect.py：Milvus 数据库管理工具，负责创建数据库、集合和索引

# 💫注意事项
    确保 Milvus 服务已启动并能通过0.0.0.0:19530访问。
    首次运行需先执行数据库初始化步骤。
    图像数据集应按目录组织，支持多级子目录。
    向量插入过程可能需要较长时间，取决于数据集大小。
    搜索结果返回最相似的 6 张图像，不足时将显示空值。

界面展示：
![项目截图](https://github.com/tanyajing1/AIGC-study/blob/78c6f2b80af36a58f4f1e15fcb4467dea9f5b45f/%E6%88%AA%E5%B1%8F2025-08-06%2017.11.47.png)







   
