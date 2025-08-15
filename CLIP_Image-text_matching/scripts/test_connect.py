# 导入Milvus相关库：连接管理、数据库操作、集合操作、数据类型定义等
from pymilvus import connections, db
from pymilvus import Collection, utility, connections, db  # 重复导入可简化为一次，此处保留原结构
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections  # 重复导入可简化为一次，此处保留原结构
import numpy as np


def crete_database():
    """
    创建Milvus数据库（注意函数名拼写应为create_database，此处保留原代码）
    操作：连接Milvus服务，创建名为"text_image_db"的数据库，并切换到该数据库
    """
    # 连接本地Milvus服务（默认端口19530）
    conn = connections.connect(host="0.0.0.0", port=19530)
    # 创建数据库"text_image_db"
    database = db.create_database("text_image_db")
    # 切换到刚创建的数据库（后续操作将在该数据库下执行）
    db.using_database("text_image_db")
    # 打印当前所有数据库，验证是否创建成功
    print(db.list_database())


def create_collection():
    """
    在"text_image_db"数据库下创建集合（Collection）
    集合用于存储图像向量、路径等数据，定义字段结构并初始化集合
    """
    # 连接Milvus服务（重复连接不影响，确保连接有效）
    conn = connections.connect(host="0.0.0.0", port=19530)
    # 切换到目标数据库
    db.using_database("text_image_db")

    # 定义集合字段：
    # 1. 主键字段m_id（唯一标识每条数据）
    m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True, )
    # 2. 图像向量字段embeding_img（存储CLIP模型生成的图像向量，维度512）
    embeding_img = FieldSchema(name="embeding_img", dtype=DataType.FLOAT_VECTOR, dim=512, )
    # 3. 图像路径字段path（存储原始图像的本地路径，字符串类型）
    path = FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256, )

    # 定义集合 schema（结构描述）
    schema = CollectionSchema(
        fields=[m_id, embeding_img, path],  # 包含上述3个字段
        description="text to image embeding search",  # 集合描述
        enable_dynamic_field=True  # 允许动态添加字段（可选）
    )

    # 创建集合（名称为"text_image_vector"）
    collection_name = "text_image_vector"
    collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
    # 打印当前连接信息，验证集合是否创建成功
    print(db.connections.list_connections())


def create_index():
    """
    为集合的向量字段创建索引（加速相似性搜索）
    索引类型为IVF_FLAT，距离度量方式为余弦相似度（COSINE）
    """
    # 连接Milvus服务
    conn = connections.connect(host="0.0.0.0", port=19530)
    # 切换到目标数据库
    db.using_database("text_image_db")

    # 定义索引参数：
    # - metric_type：距离度量方式（COSINE适合向量相似性搜索）
    # - index_type：索引类型（IVF_FLAT为基础索引，适合中小规模数据）
    # - nlist：聚类中心数量（影响搜索速度和精度，1024为常用值）
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }

    # 加载集合
    collection = Collection("text_image_vector")
    # 如果集合已有索引，先删除旧索引
    if collection.has_index():
        collection.drop_index()
    # 为向量字段"embeding_img"创建索引
    collection.create_index(
        field_name="embeding_img",
        index_params=index_params
    )

    # 打印索引构建进度（验证索引是否创建成功）
    utility.index_building_progress("text_image_vector")


def insert_data():
    """
    向集合中插入测试数据（随机生成10条向量和路径，用于验证插入功能）
    实际使用时会被insert_search.py中的真实图像数据插入替代
    """
    # 连接Milvus服务
    conn = connections.connect(host="0.0.0.0", port=19530)
    # 切换到目标数据库
    db.using_database("text_image_db")

    # 加载集合
    collection = Collection("text_image_vector")
    # 准备测试数据：m_id（主键）、随机向量、随机路径
    mids, embedings, paths = [], [], []
    data_num = 10  # 插入10条测试数据
    for idx in range(0, data_num):
        mids.append(idx)  # 主键自增
        embedings.append(np.random.normal(0, 0.1, 512).tolist())  # 随机生成512维向量
        paths.append(f'path: random num {idx}')  # 模拟图像路径

    # 插入数据到集合
    collection.insert([mids, embedings, paths])
    # 打印集合中的数据数量，验证插入是否成功
    print(collection.num_entities)


def search_collection():
    """
    测试集合的搜索功能（使用随机向量搜索相似结果，验证搜索流程）
    注意：该函数存在潜在错误（向量维度与字段不匹配、搜索字段名错误），仅用于测试
    """
    # 连接Milvus服务
    conn = connections.connect(host="0.0.0.0", port=19530)
    # 切换到目标数据库
    db.using_database("text_image_db")

    # 加载集合（搜索前需加载到内存）
    collection = Collection("text_image_vector")
    collection.load()
    # 定义搜索参数（与索引参数匹配，使用内积IP，实际应与索引的COSINE保持一致）
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    # 生成随机查询向量（注意：此处维度为768，与集合的512维不匹配，实际需修正为512）
    query_embedding = np.random.normal(0, 0.1, 768).tolist()
    # 执行搜索（注意：anns_field应为"embeding_img"，原代码"embeding"是错误的）
    results = collection.search(
        data=[query_embedding],
        anns_field="embeding",  # 错误字段名，需改为"embeding_img"
        param=search_params,
        limit=10,  # 返回前10条结果
        expr=None,
        output_fields=None,
        timeout=None,
        round_decimal=-1
    )
    # 打印搜索结果
    print(results)


def delete_collection():
    """
    删除集合（用于清理旧数据，避免重复创建导致冲突）
    若集合不存在，执行时可能报错（可忽略，不影响后续创建）
    """
    # 连接Milvus服务
    conn = connections.connect(host="0.0.0.0", port=19530)
    # 切换到目标数据库
    db.using_database("text_image_db")

    # 加载集合并删除
    collection = Collection("text_image_vector")
    collection.drop()


if __name__ == '__main__':
    # 1. 第一步：创建数据库（首次运行必须执行，解除注释）
    # crete_database()
    # 2. 第二步：删除旧集合（若存在，避免结构冲突，解除注释）
    # delete_collection()
    # 3. 第三步：创建新集合（基于定义的schema，解除注释）
    create_collection()
    # 4. 第四步：为集合创建索引（加速搜索，解除注释）
    create_index()

    # （可选）测试插入数据（验证插入功能，解除注释）
    # insert_data()
    # （可选）测试搜索功能（注意先修正函数内的错误，再解除注释）
    # search_collection()