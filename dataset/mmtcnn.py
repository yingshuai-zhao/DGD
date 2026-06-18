import os
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# ======== 路径配置 ========
input_dir = "/work/home/acxnqf2tyj/dataset/img"
output_dir = "/work/home/acxnqf2tyj/dataset/mtcnn_face"
miss_path = "missing.txt"
still_fail_txt_path = "mtcnn_failed_faces.txt"
os.makedirs(output_dir, exist_ok=True)


# ======== 每个线程维护一个独立的 MTCNN 实例 ========
thread_local = threading.local()

def get_mtcnn():
    if not hasattr(thread_local, "mtcnn"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        thread_local.mtcnn = MTCNN(thresholds=[0.1, 0.2, 0.2], keep_all=False, device=device)
    return thread_local.mtcnn

# ======== 并行处理函数（每个线程加载自己独立的模型）========
def process_image(name):
    try:
        img_path = os.path.join(input_dir, name)
        if not os.path.exists(img_path):
            return name

        img = Image.open(img_path).convert("RGB")
        mtcnn = get_mtcnn()
        boxes, _ = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            return name

        box = boxes[0].astype(int)
        x1, y1, x2, y2 = map(lambda x: max(0, x), box)
        face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)
        face_crop.save(os.path.join(output_dir, name))
        return None
    except Exception:
        return name

# ======== 主程序 ========
# 加载失败图像列表
with open(miss_path, "r") as f:
    img_list = [line.strip() for line in f if line.strip()]
still_failed = []

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(process_image, name): name for name in img_list}
    for future in tqdm(as_completed(futures), total=len(futures)):
        fail_name = future.result()
        if fail_name:
            still_failed.append(fail_name)

with open(fail_txt_path, "w") as f:
    for name in still_failed:
        f.write(name + "\n")

print(f"✅ 裁剪完成，成功: {len(img_list) - len(still_failed)}，仍失败: {len(still_failed)}")