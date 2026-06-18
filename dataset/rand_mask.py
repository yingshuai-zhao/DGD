import os
import random
import numpy as np
import cv2
from PIL import Image

def random_rectangle_mask(H=512, W=512, max_rectangles=5, max_size=0.5) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    num = random.randint(1, max_rectangles)
    for _ in range(num):
        h = int(np.random.uniform(0.1, max_size) * H)
        w = int(np.random.uniform(0.1, max_size) * W)
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        mask[y:y+h, x:x+w] = 255
    return mask

def free_form_mask(H=512, W=512, max_strokes=15, max_width=50) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    num_strokes = random.randint(1, max_strokes)
    for _ in range(num_strokes):
        start = (random.randint(0, W-1), random.randint(0, H-1))
        for _ in range(random.randint(1, 5)):
            end = (random.randint(0, W-1), random.randint(0, H-1))
            thickness = random.randint(5, max_width)
            cv2.line(mask, start, end, color=255, thickness=thickness)
            start = end
    return mask

def generate_mask(H: int, W: int) -> np.ndarray:
    if random.random() < 0.5:
        return random_rectangle_mask(H, W)
    else:
        return free_form_mask(H, W)

if __name__ == "__main__":
    # 1. 参数配置
    out_dir = "./rand_mask"
    os.makedirs(out_dir, exist_ok=True)
    H, W = 512, 512         # 掩码大小，可根据需要修改
    total = 100000            # 生成数量

    # 2. 批量生成并保存
    for i in range(total):
        mask = generate_mask(H, W)
        fname = f"mask_{i:04d}.png"
        Image.fromarray(mask).save(os.path.join(out_dir, fname))

        # 每隔 500 张打印一次进度
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{total}] masks generated")