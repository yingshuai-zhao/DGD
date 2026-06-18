import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def get_gaussian_kernel(kernel_size: int, sigma: float, device, dtype):
    # 生成 1D 高斯核
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # 生成 2D 核并归一化
    kernel2d = torch.outer(g, g)
    return kernel2d


class YCbCrBlurDataset(Dataset):
    """
    Dataset: 读取 RGB 图像文件，转为 YCbCr 张量
    返回:
        path: 图像路径
        img: Tensor, shape [3, H, W], dtype float32, 取值范围 [0,255]
    """
    def __init__(self, image_dir, extensions=('jpg', 'jpeg', 'png')):
        self.paths = []
        for fn in os.listdir(image_dir):
            if fn.lower().endswith(extensions):
                self.paths.append(os.path.join(image_dir, fn))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).resize((512,512))
        img =img.convert('YCbCr')  # 转 YCbCr
        arr = np.array(img, dtype=np.uint8)      # (H, W, 3)
        # 转 Tensor 并调整维度
        tensor = torch.from_numpy(arr).float().permute(2, 0, 1)  # [3, H, W]
        return path, tensor


def process_and_save(batch, kernel, output_dir, device):
    paths, imgs = batch
    imgs = imgs.to(device)  # [B,3,H,W]
    # 分离通道
    y, cb, cr = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
    # 准备卷积核
    k = kernel.shape[0]
    kernel_tensor = kernel.to(device, imgs.dtype).view(1, 1, k, k)
    # 对 Y 通道做高斯模糊
    y_blur = F.conv2d(y, kernel_tensor, padding=k//2)
    # 拼回并 clamp
    out = torch.cat([y_blur, cb, cr], dim=1).clamp(0, 255)

    # 转回 RGB 并保存
    for i, path in enumerate(paths):
        ycbcr = out[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_pil = Image.fromarray(ycbcr, mode='YCbCr').convert('RGB')
        # img_pil=img_pil.resize((512,512))
        fn = os.path.basename(path)
        img_pil.save(os.path.join(output_dir, fn))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/root/dataset/img',help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='/root/dataset/blur', help='输出图像目录')
    parser.add_argument('--kernel_size', type=int, default=59)
    parser.add_argument('--sigma', type=float, default=150.0)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成高斯核
    kernel = get_gaussian_kernel(args.kernel_size, args.sigma, device, torch.float32)

    # 数据集和 DataLoader
    dataset = YCbCrBlurDataset(args.input_dir)
    def collate_fn(batch):
        paths, tensors = zip(*batch)
        imgs = torch.stack(tensors, dim=0)
        return list(paths), imgs

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # 批处理并保存
    for batch in dataloader:
        process_and_save(batch, kernel, args.output_dir, device)
        print(1)

    print('处理完成')
