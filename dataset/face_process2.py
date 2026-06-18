import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def estimate_norm(landmarks, image_size=112):
    arcface_dst = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32
    )
    ratio = float(image_size) / 112.0
    dst = arcface_dst * ratio
    tform = cv2.estimateAffinePartial2D(landmarks, dst)[0]
    return tform

def process_image(app, image_path, output_path, error_log_path, image_size=112):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        with open(error_log_path, 'a') as f:
            f.write(os.path.basename(image_path) + '\n')
        return False

    face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    landmarks = face.kps
    M = estimate_norm(landmarks, image_size)
    aligned_face = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    cv2.imwrite(output_path, aligned_face)
    return True

# ---------- 主逻辑 ----------
if __name__ == '__main__':
    input_folder = 'img'        # 输入图像目录
    output_folder = 'face'      # 对齐图像保存目录
    error_txt = 'error.txt'     # 第一轮失败图像列表
    final_error_txt = 'error_final.txt'  # 第二轮仍失败的图像

    # 初始化低阈值检测模型
    app = FaceAnalysis(name='antelopev2', root='/root/fangan',
                       providers=['CUDAExecutionProvider'],
                       allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_thresh=0.01, det_size=(512, 512))

    # 清空最终失败记录
    open(final_error_txt, 'w').close()

    # 读取失败图像列表并重新处理
    with open(error_txt, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    for filename in filenames:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f"[Retry] {filename}")
        success = process_image(app, input_path, output_path, final_error_txt)
        if success:
            print(f"✅ Saved aligned face: {output_path}")
        else:
            print(f"❌ Still failed: {filename}")