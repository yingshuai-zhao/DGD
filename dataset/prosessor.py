import os
from PIL import Image
import numpy as np

# 源文件夹和目标文件夹路径
source_folder = "/root/dataset/img"
target_folder = "/root/celeba"

# 确保目标目录存在
os.makedirs(target_folder, exist_ok=True)

for i in range(30000):  # 0 ~ 69999
    source_filename = f"{i:05d}.jpg"
    source_path = os.path.join(source_folder, source_filename)

    if not os.path.exists(source_path):
        print(f"跳过不存在的文件: {source_path}")
        continue


    # 保存新图像
    new_filename = f"{i:05d}.jpg"
    target_path = os.path.join(target_folder, new_filename)
    # 打开并重新保存（不改变内容）
    Image.open(source_path).save(target_path)

    if i % 5000 == 0:
        print(f"处理到第 {i} 张图像，保存为 {new_filename}")


