import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid
import torchvision
from torchvision.utils import save_image
# 可选：感知损失和 ID 特征提取模型
# from lpips import LPIPS
from PIL import Image
from torchvision import transforms
import os

# lpips_fn = LPIPS(net='vgg').eval().cuda()
tensor_112 = transforms.Resize((112,112), interpolation=transforms.InterpolationMode.BILINEAR)


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

    lambda_cos=0.05
    lambda_euclidean=0.001
    idloss1=(lambda_cos * cos_loss).mean()
    idloss2=(lambda_euclidean/euclidean_loss).mean()
    loss += idloss1+idloss2
    return loss,idloss1,idloss2