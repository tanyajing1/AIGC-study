import time
from matplotlib import pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import warnings
import os
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")

# 模型名称
model_name = "openai/clip-vit-large-patch14"
# 定义当前目录
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, model_name.replace("/", "_"))

# 检查当前目录是否有预训练权重文件，如果没有则下载
def download_pretrained_weights_if_needed(model_name, save_dir):
    if not os.path.exists(save_dir):
        try:
            print(f"Downloading {model_name} to {save_dir}...")
            snapshot_download(repo_id=model_name, local_dir=save_dir, local_dir_use_symlinks=False)
            print(f"{model_name} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")

download_pretrained_weights_if_needed(model_name, model_dir)

# 加载模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 函数：生成文本嵌入
def text_embedding(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.cpu().numpy()

def get_image_embeddings(image_paths):
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"Error loading image {e}")
    if not images:
        return None
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1, axis=1, keepdims=True) * np.linalg.norm(vec2, axis=1))

# 递归遍历目录获取所有图片路径
def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in ['.png', '.jpg', '.jpeg']:
                image_paths.append(os.path.join(root, file))
    return image_paths

# 获取候选分类名列表
def get_candidates(directory):
    candidates = []
    for sub_dir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, sub_dir)
        if os.path.isdir(sub_dir_path):
            candidates.append(f"a photo of {sub_dir}")
    return candidates

# 测试图片分类正确率
def accuracy(image_paths, candidates, text_embeddings, batch_size=64):
    correct_count = 0
    total_count = len(image_paths)
    num_batches = (total_count + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_count)
        batch_image_paths = image_paths[start_idx:end_idx]

        image_embeddings = get_image_embeddings(batch_image_paths)
        if image_embeddings is not None:
            similarities = cosine_similarity(image_embeddings, text_embeddings)
            predicted_indices = np.argmax(similarities, axis=1)

            for j, predicted_index in enumerate(predicted_indices):
                predicted_category = candidates[predicted_index].split(" ")[-1]
                actual_category = os.path.basename(os.path.dirname(batch_image_paths[j]))
                if predicted_category == actual_category:
                    correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

# 图片分类
def flowerClassify():
    image_paths = get_all_image_paths("./flower_photos")
    candidates = get_candidates("./flower_photos")
    text_embeddings = text_embedding(candidates)
    start_time = time.time()
    acc = accuracy(image_paths, candidates, text_embeddings, batch_size=64)
    end_time = time.time()
    print(f"Time taken to test accuracy: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {acc * 100:.2f}%")


##################################################################################################3

# 遍历 data 目录获取所有图片路径
def get_images_from_data_dir():
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []
    return get_all_image_paths(data_dir)


# 找到与文本最匹配的图片
def find_most_matching_image(text, image_paths):
    text_emb = text_embedding([text])
    image_embeddings = get_image_embeddings(image_paths)
    if image_embeddings is None:
        return None
    similarities = cosine_similarity(text_emb, image_embeddings)
    most_matching_index = np.argmax(similarities)
    return image_paths[most_matching_index]


# 根据文字搜索图片
def searchPicByText():
    image_paths = get_images_from_data_dir()
    query_text = "a photo of a sunflowers"
    most_matching_image = find_most_matching_image(query_text, image_paths)
    if most_matching_image:
        print(f"The most matching image for '{query_text}' is: {most_matching_image}")
        try:
            img = Image.open(most_matching_image)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Most matching image for '{query_text}'")
            plt.show()
        except Exception as e:
            print(f"Error opening image: {e}")
    else:
        print("No matching image found.")
