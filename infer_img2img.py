import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms import InterpolationMode
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import AutoencoderKL,UniPCMultistepScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import insightface_backbone as model_insightface_backbone
from inpaint_pipeline import StableDiffusionanonImg2ImgPipeline

class InferenceDataset(Dataset):
    def __init__(self, img_dir, parse_dir, normal_dir,blur_dir, start_idx, end_idx, face_transform):
        self.img_dir    = img_dir
        self.parse_dir  = parse_dir
        self.normal_dir = normal_dir
        self.blur_dir = blur_dir
        self.indices    = list(range(start_idx, end_idx))
        self.face_tf    = face_transform
        self.to_tensor  = transforms.PILToTensor()
        self.resize512  = transforms.Resize((512,512), interpolation=InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # 1) 原图
        image = load_image(f"{self.img_dir}/{i:05d}.jpg").convert("RGB")
        # 2) face crop tensor (CPU)
        face_img = self.face_tf(image)  # Tensor [3,128,128], CPU
        # image = load_image(f"{self.img_dir}/{i:05d}.jpg").convert("RGB")
        # 3) 法线图
        normal_img = Image.open(f"{self.normal_dir}/{i:05d}.jpg").convert("RGB")
        blur_img = Image.open(f"{self.blur_dir}/{i:05d}.jpg").convert("RGB")
        # 4) parse 掩码
        parse = Image.open(f"{self.parse_dir}/{i:05d}.png").convert("L")
        mask   = self.to_tensor(parse)            # [1,H,W], uint8
        mask   = self.resize512(mask) / 255.0     # [1,512,512], float
        mask   = ((mask > 5/255.) & (mask < 125/255.)).float()  # 0/1

        return {
            "index":      i,
            "image":      image,       # PIL
            "face_tensor":face_img,    # CPU tensor
            "mask":       mask,        # CPU tensor
            "control":    normal_img,  # PIL
            "blur_img":   blur_img,
        }
def collate_fn(batch):
    indices   = [b["index"] for b in batch]
    images    = [b["image"] for b in batch]
    blur_imgs    = [b["blur_img"] for b in batch]
    controls  = [b["control"] for b in batch]
    faces     = torch.stack([b["face_tensor"] for b in batch], dim=0)  # [B,3,128,128]
    masks     = torch.stack([b["mask"] for b in batch], dim=0)        # [B,1,512,512]
    return indices, images, faces, masks, controls,blur_imgs
if __name__ == "__main__":
    # —— 初始化 ID encoder —— 

    device = "cuda"
    ID_encoder = model_insightface_backbone.getarcface(
        'path/to/your/pretrained/insightface_glint360k.pth'
    )
    ID_encoder.to(device).eval()
    ID_encoder.requires_grad_(False)
    # —— 初始化 ControlNet 管道 —— 
    controlnet = ControlNetModel.from_pretrained(
        './stage1/base/controlnet', torch_dtype=torch.float16
    )
    pipe = StableDiffusionanonImg2ImgPipeline.from_pretrained(
        'path/to/your/stable_diffusion_1.5',
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_anon(
        './ruohua_all/checkpoint-180000/pytorch_model.bin',
        './qianghua/checkpoint-200000/pytorch_model.bin',
        CLIPVisionModelWithProjection.from_pretrained('path/to/your/ipa/models/image_encoder')
    )
    pipe.cuda()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = KarrasDiffusionSchedulers.from_config(pipe.scheduler.config)

    # —— 定义 transform —— 
    face_transform = transforms.Compose([
        transforms.Resize((128,128), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])

    # —— 构建 DataLoader —— 
    dataset = InferenceDataset(
        img_dir='./dataset/img',
        parse_dir='./dataset/parse',
        normal_dir='./dataset/face_pose',
        blur_dir='./dataset/blur', 
        start_idx=0,
        end_idx=30000,
        face_transform=face_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,
        num_workers=5,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    save_path = 'output'

    os.makedirs(save_path, exist_ok=True)

    # —— 批量推理并保存 —— 
    with torch.no_grad():
        for indices, images, faces, masks, controls,blur_imgs in loader:
            feature_embs = ID_encoder(faces.to('cuda'),return_id512=False)
            outs = pipe(
                image=images,
                blur_img=blur_imgs,
                feature_embeds=feature_embs,      # [B, tokens, dim]
                mask_image=masks,                # [B,1,512,512]
                control_image=controls,          # list[PIL]
                controlnet_conditioning_scale=0.4,
                num_inference_steps=30,
                guidance_scale=1.7,
                strength=0.9,
            ).images  # list of PIL

            for idx, img_out in zip(indices, outs):
                img_out.save(f'{save_path}/{idx:05d}.jpg')

            print(f"Processed batch up to index {indices[-1]:05d}")