import os
import cv2
import dlib
from multiprocessing import Pool, cpu_count
# 输入输出路径
input_dir = "img"
output_dir = "face"
fail_txt_path = "failed_faces.txt"

os.makedirs(output_dir, exist_ok=True)

detector = dlib.get_frontal_face_detector()


def process_image(i):
    img_name = f"{i:05d}.jpg"
    img_path = os.path.join(input_dir, img_name)

    if not os.path.exists(img_path):
        return img_name  # 作为失败项返回

    img = cv2.imread(img_path)
    if img is None:
        return img_name

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return img_name

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    x, y = max(x, 0), max(y, 0)

    try:
        cropped = img[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (224, 224))
        cv2.imwrite(os.path.join(output_dir, img_name), resized)
        return None  # 成功
    except:
        return img_name  # 失败

if __name__ == "__main__":
    total_images = 100000
    with Pool(processes=6) as pool:  # 使用6个进程
        failed_files = pool.map(process_image, range(total_images))

    # 过滤掉 None 的成功项
    failed_files = [f for f in failed_files if f is not None]

    # 保存失败列表
    with open(fail_txt_path, "w") as f:
        for name in failed_files:
            f.write(name + "\n")

    print(f"处理完成，失败数：{len(failed_files)}，保存于 {fail_txt_path}")