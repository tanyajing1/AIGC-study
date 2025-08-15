# 导入必要的库
import os
import torch
from torch import nn
from PIL import Image
import yaml
from transformers import CLIPProcessor, CLIPModel
from pymilvus import Collection, connections, db

# 全局变量声明
clip_model = None
processor = None
collection = None  # Milvus集合全局变量


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def init_model():
    """初始化CLIP模型和处理器（确保全局可用）"""
    global clip_model, processor
    if clip_model is None or processor is None:
        cfg = load_config('../cfg/config.yaml')
        # 从配置文件加载模型
        clip_model = CLIPModel.from_pretrained(cfg["model"])
        processor = CLIPProcessor.from_pretrained(cfg["model"])


def init_milvus():
    """初始化Milvus连接和集合（确保全局可用）"""
    global collection
    if collection is None:
        # 连接Milvus服务
        connections.connect(host="0.0.0.0", port=19530)
        # 切换到目标数据库
        db.using_database("text_image_db")
        # 加载集合
        collection = Collection("text_image_vector")
        collection.load()  # 搜索前必须加载到内存


def embeding_text(text):
    """将文本转换为向量"""
    init_model()  # 确保模型已初始化
    text_inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_inputs)
    return text_embeddings.flatten().detach().numpy().tolist()


def embeding_img(img_data):
    """将图像转换为向量"""
    init_model()  # 确保模型已初始化
    img_embeding = processor(images=img_data, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**img_embeding)
    return image_embeddings.flatten().detach().numpy().tolist()


def update_image_vector(data_path, operator):
    """批量插入图像向量到Milvus"""
    idxs, embedings, paths = [], [], []
    total_count = 0
    try:
        for dir_name in os.listdir(data_path):
            sub_dir = os.path.join(data_path, dir_name)
            if not os.path.isdir(sub_dir):
                continue
            for file in os.listdir(sub_dir):
                if not file.endswith('.jpg'):
                    continue
                print(f'{dir_name}/{file} is processing')
                image = Image.open(os.path.join(sub_dir, file))
                embeding = embeding_img(image)

                idxs.append(total_count)
                embedings.append(embeding)
                paths.append(os.path.join(sub_dir, file))
                total_count += 1

                if total_count % 50 == 0:
                    data = [idxs, embedings, paths]
                    operator.insert(data)
                    print(f'success insert {total_count} items')
                    idxs, embedings, paths = [], [], []

            if len(idxs):
                data = [idxs, embedings, paths]
                operator.insert(data)
                print(f'success insert {total_count} items')
    except Exception as e:
        print(e)
    print(f'finish update items: {total_count}')


def search_data(input_embeding):
    """在Milvus中搜索相似向量"""
    init_milvus()  # 确保Milvus已连接
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[input_embeding],
        anns_field="embeding_img",
        param=search_params,
        limit=6,
        output_fields=["path"]
    )
    img_paths = [hit.entity.get("path") for hit in results[0]]
    return img_paths


def image_search(text, img_input):
    """处理搜索请求（文本或图像输入）"""
    if text == '' and img_input is None:
        return [None] * 6  # 返回空结果（与输出数量匹配）

    # 生成查询向量
    if text != '':
        input_embeding = embeding_text(text)
    elif img_input is not None:
        input_embeding = embeding_img(img_input)
    else:
        return [None] * 6

    # 搜索并返回结果
    results = search_data(input_embeding)
    # 将路径转换为PIL图像（Gradio需要直接显示图像）
    output_images = []
    for path in results:
        if path and os.path.exists(path):
            output_images.append(Image.open(path))
        else:
            output_images.append(None)  # 路径不存在时返回空
    return output_images


if __name__ == '__main__':
    """批量插入图像向量的入口"""
    cfg = load_config('../cfg/config.yaml')
    data_dir = cfg['data_path']

    # 初始化模型和Milvus
    init_model()
    init_milvus()

    # 执行批量插入
    update_image_vector(data_dir, collection)