import os
import cv2
import dlib
from multiprocessing import Pool, cpu_count

# 路径配置
input_dir = "/work/home/acxnqf2tyj/dataset/img"
retry_output_dir = "/work/home/acxnqf2tyj/dataset/face2"
fail_txt_path = "failed_faces.txt"
still_fail_txt_path = "still_failed_faces.txt"

os.makedirs(retry_output_dir, exist_ok=True)

# 加载失败图像列表
with open(fail_txt_path, "r") as f:
    failed_list = [line.strip() for line in f if line.strip()]

# 初始化 dlib 检测器（需要在每个子进程中初始化）
def init_detector():
    global cnn_detector
    cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

def process_image(name):
    try:
        img_path = os.path.join(input_dir, name)
        if not os.path.exists(img_path):
            return name  # 失败

        img = cv2.imread(img_path)
        if img is None:
            return name

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = cnn_detector(gray, 1)
        valid_faces = [d.rect for d in detections if d.confidence > 0.1]

        if len(valid_faces) == 0:
            return name

        face = max(valid_faces, key=lambda r: r.width() * r.height())
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y = max(x, 0), max(y, 0)
        cropped = img[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (224, 224))
        cv2.imwrite(os.path.join(retry_output_dir, name), resized)
        return None  # 成功
    except Exception as e:
        print(f"异常 {name}: {e}")
        return name

if __name__ == "__main__":
    print(f"使用 {min(3, cpu_count())} 个进程进行人脸重试检测中...")
    with Pool(processes=min(3, cpu_count()), initializer=init_detector) as pool:
        still_failed = pool.map(process_image, failed_list)

    # 过滤出失败的文件名
    still_failed = [name for name in still_failed if name is not None]

    with open(still_fail_txt_path, "w") as f:
        for name in still_failed:
            f.write(name + "\n")

    print(f"✅ 成功提取: {len(failed_list) - len(still_failed)}，仍失败: {len(still_failed)}")