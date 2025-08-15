import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import torch
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_model_cache_path(model_id):
    """获取模型在本地缓存中的真实路径"""
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"

    if not snapshots_dir.exists():
        return None

    # 获取最新的snapshot（按创建时间排序）
    snapshots = sorted(snapshots_dir.iterdir(), key=os.path.getmtime, reverse=True)
    return snapshots[0] if snapshots else None


# 模型ID
model_id = "openai/clip-vit-base-patch32"

# 检查模型是否已缓存
cache_path = get_model_cache_path(model_id)
if cache_path:
    print(f"模型已缓存，路径为: {cache_path}")
else:
    print("模型未缓存，开始下载...")

# 加载模型（自动使用缓存）
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
clip_processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
model = CLIPModel.from_pretrained(model_id)
print("模型加载完成")

# 示例文本描述
caption = "A dog runs on grass"

# 示例图片URL（也可以使用本地路径）
image_path = "/Users/annie/Desktop/截屏2025-08-01 15.37.34.png"
try:
    image = Image.open(image_path).convert("RGB")  # 确保转换为RGB格式
    print("本地图片加载成功！")
except Exception as e:
    print(f"图片加载失败: {e}")
    exit()

# 2-处理文本
# 分词并生成文本嵌入
inputs = clip_tokenizer(caption, return_tensors="pt")  # Fixed syntax error
text_embedding = model.get_text_features(**inputs)

# 3-处理图片
processed_image = clip_processor(
    text=None,
    images=image,
    return_tensors='pt'
)['pixel_values']

# 生成图片嵌入
image_embedding = model.get_image_features(processed_image)

# 4-嵌入向量归一化
text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

# 5-计算相似度
score = (text_embedding @ image_embedding.T)[0][0].item()  # 提取标量值
print(f"文本描述: '{caption}'")
print(f"图片路径: {image_path}")
print(f"相似度分数: {score:.4f}")

# 显示图片和结果
plt.imshow(image)
plt.title(f"文本: '{caption}'\n相似度: {score:.4f}")
plt.axis('off')
plt.show()