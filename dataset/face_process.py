import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis

# 只加载检测模型
app = FaceAnalysis(name='antelopev2', root='/root/fangan',
                   providers=['CUDAExecutionProvider'],
                   allowed_modules=['detection'])
app.prepare(ctx_id=0,  det_thresh=0.01,det_size=(512, 512))

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

def process_image(image_path, output_path, error_log_path, image_size=112):
    img = cv2.imread(image_path)
    faces = app.get(img)

    # 如果没有检测到人脸，记录到 error.txt
    if not faces or len(faces) == 0:
        with open(error_log_path, 'a+') as f:
            f.write(os.path.basename(image_path) + '\n')
        return

    # 选择最大人脸
    face = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    landmarks = face.kps
    M = estimate_norm(landmarks, image_size)
    aligned_face = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    cv2.imwrite(output_path, aligned_face)

def process_images_in_folder(input_folder, output_folder, image_size=112):
    error_log_path = 'error.txt'
    # 清空之前的错误记录
    open(error_log_path, 'w').close()

    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # print(f'Processing {input_path}...')
            process_image(input_path, output_path, error_log_path, image_size)
            # print(f'Saved aligned face to {output_path}')

if __name__ == '__main__':
    input_folder = 'img'
    output_dir = 'face'
    os.makedirs(output_dir, exist_ok=True)
    process_images_in_folder(input_folder, output_dir)
    print("Batch processing completed!")