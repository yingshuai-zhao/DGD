import os
import random
from PIL import Image
import numpy as np

def pil_to_tensor(pil_image):
    """将PIL图像转换为numpy数组"""
    return np.array(pil_image)

# 假设我们有一张图片路径
image_path = "/root/dataset/parse/00001.png"  # 替换为你的图片路径
output_mask_path = "mask.png"  # 保存mask的路径

# 加载图片并转换为灰度
parse_image = Image.open(image_path).convert("L")

# 转换为numpy数组
mask_u8 = pil_to_tensor(parse_image)

# 生成候选值并随机选择
candidates = list(range(20, 125, 10))
selected_vals = random.sample(candidates, k=random.randint(4, 11))

# 初始化mask
mask = ((mask_u8 >= 6) & (mask_u8 <= 14))

# 添加随机选择的区间
for v in selected_vals:
    lower = v - 4
    upper = v + 4
    submask = (mask_u8 >= lower) & (mask_u8 <= upper)
    mask |= submask  # 合并

# 将布尔mask转换为0-255的uint8图像
mask_u8 = mask.astype(np.uint8) * 255

# 保存mask
mask_image = Image.fromarray(mask_u8)
mask_image.save(output_mask_path)

print(f"Mask saved to {output_mask_path}")