import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid
import torchvision
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import os

# lpips_fn = LPIPS(net='vgg').eval().cuda()
tensor_112 = transforms.Resize((112,112), interpolation=transforms.InterpolationMode.BILINEAR)


def save_images_as_pil(images_tensor, output_dir, prefix="image"):
    """
    将图像张量保存为PIL格式图片
    Args:
        images_tensor (torch.Tensor): 输入的图像张量，形状 (B, 3, 512, 512)
        output_dir (str): 输出目录
        prefix (str): 保存的文件名前缀
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        # 将图像张量转换为 [B, 3, 512, 512] 格式的 PIL 图像并保存
        batch_size = images_tensor.shape[0]
        img_tensor = images_tensor[0].cpu()  # 确保在CPU上
        img_tensor = (img_tensor + 1) / 2.0  # 反归一化到 [0, 1] 范围

        # 将张量转为 [H, W, C] 格式，并转换为 PIL 图像
        img = img_tensor.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转到 (H, W, C)
        pil_img = Image.fromarray((img * 255).astype('uint8'))  # 转换为 PIL 图像并恢复为 [0, 255] 范围

        # 保存图像
        pil_img.save(os.path.join(output_dir, f"{prefix}0.jpg"))

def compute_image_loss(
    noise_pred,
    latents,                  # (B, 4, H, W)
    timesteps,                # (B,)
    vae,                      # VAE decoder
    scheduler,                # DDIM or DPM++ scheduler
    x0,                       # (B, 3, 512, 512)
    src_ID=None,
    id_model=None,
    device = 'cuda',
):

    z0_hat_list = []
    for i in range(noise_pred.shape[0]):
        t = timesteps[i].item()
        lat = latents[i].unsqueeze(0)
        pred = noise_pred[i].unsqueeze(0)

        step_output = scheduler.step(
            model_output=pred,
            timestep=t,
            sample=lat,
            return_dict=True
        )

        z0_hat_list.append(step_output.pred_original_sample)
    x0=x0.to(device)
    z0_hat = torch.cat(z0_hat_list, dim=0)  # shape: [B, ...]
    # Step 3: 解码 z0_hat 成图像
    x0_hat = vae.decode(z0_hat.to(dtype=vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]  # (B, 3, 512, 512)
    # 初始化总 loss
    loss = 0.0
    # ===================== ID Loss =====================
    id_out = id_model(tensor_112(x0_hat),return_id512=True)
    cos_loss = F.cosine_similarity(src_ID, id_out, dim=1)
    # arcos_loss = torch.acos(cos_loss)  # (bs,)
    euclidean_loss = F.pairwise_distance(src_ID, id_out) # (bs,)

    lambda_cos=0.2
    lambda_euclidean=0.0005
    idloss1=(lambda_cos * (1-cos_loss)).mean()
    idloss2=(lambda_euclidean * euclidean_loss).mean()
    loss += idloss1+idloss2
    return loss,idloss1,idloss2