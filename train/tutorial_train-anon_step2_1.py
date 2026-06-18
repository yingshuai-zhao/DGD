import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import cv2
from datetime import datetime
import numpy as np
from sympy import im
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from loss.loss2_1 import compute_image_loss
from diffusers import DDIMScheduler
from torchvision.transforms.functional import to_pil_image
from accelerate.utils import DistributedDataParallelKwargs
import insightface_backbone as model_insightface_backbone
from time_faceid_mlp import ID2Token,Image2Token
import math
import shutil

import bitsandbytes as bnb
# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, size=512,id_drop_rate=0.05,it_drop_rate=0.3, image_root_path="",):
        super().__init__()

        self.size = size
        self.id_drop_rate = id_drop_rate
        self.it_drop_rate = it_drop_rate
        self.image_root_path = '/data/dataset_r/dataset'
        # 获取 img 文件夹中所有文件名并排序
        self.img_dir = os.path.join(self.image_root_path, 'img')
        self.img_data = sorted(os.listdir(self.img_dir))

        # 获取 parse 文件夹中所有文件名并排序
        self.parse_dir = os.path.join(self.image_root_path, 'parse')
        self.parse_data = sorted(os.listdir(self.parse_dir))

        self.rand_mask_dir = os.path.join(self.image_root_path, 'rand_mask')
        self.rand_mask_data = sorted(os.listdir(self.rand_mask_dir))

        self.normal_dir = os.path.join(self.image_root_path, 'face_pose')
        self.normal_data = sorted(os.listdir(self.normal_dir))
        
        self.blur_dir = os.path.join(self.image_root_path, 'blur')
        self.blur_data = sorted(os.listdir(self.blur_dir))
        
        assert len(self.parse_data) == len(self.img_data)
        assert len(self.normal_data) == len(self.img_data)
        assert len(self.blur_data) == len(self.img_data)
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.pil_to_tensor = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        )
        self.face_transform = transforms.Compose([
            transforms.Resize((128,128), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.controlnet_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.clip_image_processor = CLIPImageProcessor()

        
    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.img_dir, self.img_data[idx]) 
        parse_image_path = os.path.join(self.parse_dir, self.parse_data[idx]) 
        normal_image_path = os.path.join(self.normal_dir, self.normal_data[idx]) 
        blur_image_path = os.path.join(self.blur_dir, self.blur_data[idx]) 
        rand_mask_path = os.path.join(self.rand_mask_dir, self.rand_mask_data[random.randint(0, 99999)]) 
        
        image = Image.open(raw_image_path).convert("RGB")
        normal_image = Image.open(normal_image_path).convert("RGB")
        blur_image = Image.open(blur_image_path).convert("RGB")
        face_image = self.face_transform(image)
        image=image.resize((512,512),resample=Image.BILINEAR)
        is_parse=0
        if random.random() < 0.4:
            is_parse=1
            parse_image = Image.open(parse_image_path).convert("L")
        else:
            is_parse=0
            mask=Image.open(rand_mask_path).convert("L")

        # # 同步数据增强：左右翻转
        # if random.random() < 0.5:
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #     if is_parse==1:
        #         parse_image = parse_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 同步数据增强：随机裁剪
        if random.random() < 0.4:
            i_width, i_height = image.size
            crop_ratio = random.uniform(0.8, 1.0) 
            crop_size = int(self.size * crop_ratio)
            left = random.randint(0, i_width - crop_size)
            top = random.randint(0, i_height - crop_size)
            right = left + crop_size
            bottom = top + crop_size
            image = image.crop((left, top, right, bottom))
            blur_image = blur_image.crop((left, top, right, bottom))
            if is_parse==1:
                parse_image = parse_image.crop((left, top, right, bottom))
                parse_image=parse_image.resize((512,512),resample=Image.BILINEAR)
            image=image.resize((512,512),resample=Image.BILINEAR)
            blur_image=blur_image.resize((512,512),resample=Image.BILINEAR)
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        if is_parse==1:
            mask_u8 = self.pil_to_tensor(parse_image)
            candidates = list(range(20, 125, 10))
            selected_vals = random.sample(candidates, k=random.randint(4, 11))
            mask = ((mask_u8 >= 6) & (mask_u8 <= 14))
            for v in selected_vals:
                lower = v - 4
                upper = v + 4
                submask = (mask_u8 >= lower) & (mask_u8 <= upper)
                mask |= submask  # 合并
            mask=mask.float()
        else:
            mask=self.controlnet_transform(mask)

        normal_image=self.controlnet_transform(normal_image)
        image0 = self.transform(image)
        masked_image=image0 * (mask < 0.5)
        clip_image = self.clip_image_processor(images=blur_image, return_tensors="pt").pixel_values

        # 是否 drop image embedding
        drop_id_embed = int(random.random() < self.id_drop_rate)
        drop_it_embed = int(random.random() < self.it_drop_rate)
        return {
            "face_image": face_image,
            "mask":mask,
            "image": image0,
            "masked_image":masked_image,
            "normal_image": normal_image,
            "clip_image": clip_image,
            "drop_id_embed": drop_id_embed,
            "drop_it_embed": drop_it_embed,
        }
    def __len__(self):
        return len(self.img_data)
    

def collate_fn(data):
    face_images = torch.stack([example["face_image"] for example in data])
    masks = torch.stack([example["mask"] for example in data])
    masked_images = torch.stack([example["masked_image"] for example in data])
    images = torch.stack([example["image"] for example in data])
    normal_images = torch.stack([example["normal_image"] for example in data])
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_id_embeds = [example["drop_id_embed"] for example in data]
    drop_it_embeds = [example["drop_it_embed"] for example in data]

    return {
        "face_images":face_images,
        "masks":masks,
        "images": images,
        "masked_images":masked_images,
        "normal_images": normal_images,
        "clip_images": clip_images,
        "drop_id_embeds": drop_id_embeds,
        "drop_it_embeds": drop_it_embeds,
    }
    

class AnonAdapter(torch.nn.Module):
    def __init__(self, unet,controlnet, image_proj_model, feature_proj_model):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.image_proj_model = image_proj_model
        self.feature_proj_model = feature_proj_model


    def forward(self,noisy_latents, mask, masked_images, timesteps,image_embeds,id_embeds,controlnet_image):
        # print(image_embeds.shape) 8*1024
        # assert 0
        img_tokens = self.image_proj_model(image_embeds)
        id_tokens = self.feature_proj_model(id_embeds, timesteps)
        # bs*77*768
        encoder_hidden_states_unet = id_tokens
        encoder_hidden_states_controlnet = img_tokens
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states_controlnet,  # Insightface feature
            controlnet_cond=controlnet_image,  # keypoints image
            return_dict=False,
        )
        latent_model_input = torch.cat([noisy_latents, mask, masked_images], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states_unet,
            down_block_additional_residuals=[sample for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        return noise_pred


    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/data/sd15',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default='/data/fangan4/ipa/models/image_encoder',
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )
    
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("/root/vae_sme")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    controlnet = ControlNetModel.from_pretrained('./wo_loss/checkpoint-120000/controlnet')

    ID_encoder = model_insightface_backbone.getarcface('./pretrained/insightface_glint360k.pth').to(accelerator.device)
    ID_encoder.eval()

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    ID_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    # controlnet.train()
    
    feature_proj_model = ID2Token(id_dim=512, text_hidden_size=768, max_length=77, num_layers=3)

    image_proj_model = Image2Token(visual_hidden_size=image_encoder.config.projection_dim,text_hidden_size=unet.config.cross_attention_dim,num_layers=3)
    image_proj_model.requires_grad_(False)
    anon_adapter = AnonAdapter(unet, controlnet, image_proj_model,feature_proj_model)

    tensor_112 = transforms.Resize((112,112), interpolation=transforms.InterpolationMode.BILINEAR)
    tensor_64 = transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR)
    # Register a hook function to process the state of a specific module before saving.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model_instance in enumerate(models):
                if isinstance(model_instance, AnonAdapter):
                    # When saving a checkpoint, only save the anon-adapter and image_proj, do not save the unet.
                    anon_adapter_state = {
                        'image_proj': model_instance.image_proj_model.state_dict(),
                        'feature_proj': model_instance.feature_proj_model.state_dict(),
                    }
                    torch.save(anon_adapter_state, os.path.join(output_dir, 'pytorch_model.bin'))
                    print(f"Anon-Adapter Model weights saved in {os.path.join(output_dir, 'pytorch_model.bin')}")
                    # Save controlnet separately.
                    # sub_dir = "controlnet"
                    # model_instance.controlnet.save_pretrained(os.path.join(output_dir, sub_dir))
                    # print(f"Controlnet weights saved in {os.path.join(output_dir, sub_dir)}")
                    weights.pop(i)
                    break

    def load_model_hook(models, input_dir):
        # find instance of AnonAdapter Model.
        while len(models) > 0:
            model_instance = models.pop()
            if isinstance(model_instance, AnonAdapter):
                anon_adapter_path = os.path.join(input_dir, 'pytorch_model.bin')
                if os.path.exists(anon_adapter_path):
                    anon_adapter_state = torch.load(anon_adapter_path)
                    model_instance.image_proj_model.load_state_dict(anon_adapter_state['image_proj'])
                    model_instance.feature_proj_model.load_state_dict(anon_adapter_state['feature_proj'])
                    # sub_dir = "controlnet"
                    # model_instance.controlnet.from_pretrained(os.path.join(input_dir, sub_dir))
                    print(f"Model weights loaded from {anon_adapter_path}")
                else:
                    print(f"No saved weights found at {anon_adapter_path}")


    # Register hook functions for saving  and loading.
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    ID_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device)   



    # optimizer
    params_to_opt = itertools.chain(anon_adapter.image_proj_model.parameters(),
                                anon_adapter.feature_proj_model.parameters(),
                                anon_adapter.controlnet.parameters())
    optimizer = bnb.optim.AdamW8bit(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    anon_adapter, optimizer, train_dataloader = accelerator.prepare(anon_adapter, optimizer, train_dataloader)

    # Restore checkpoints
    checkpoint_folders = [folder for folder in os.listdir(args.output_dir) if folder.startswith('checkpoint-')]
    if checkpoint_folders:
        # Extract step numbers from all checkpoints and find the maximum step number
        global_step = max(int(folder.split('-')[-1]) for folder in checkpoint_folders if folder.split('-')[-1].isdigit())
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        # Load the checkpoint
        accelerator.load_state(checkpoint_path)
    else:
        global_step = 0
        print("No checkpoint folders found.")


    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(anon_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    masked_images = vae.encode(batch["masked_images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    masked_images = masked_images * vae.config.scaling_factor
                    rich_ID = ID_encoder(batch["face_images"].to(accelerator.device, dtype=weight_dtype),return_id512=False)
                    src_poor_ID=ID_encoder(tensor_112(batch["images"]).to(accelerator.device, dtype=weight_dtype),return_id512=True)
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    masks = batch["masks"]
                    mask = tensor_64(masks).to(accelerator.device, dtype=weight_dtype)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)                
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                id_embeds_= []
                for id_embed, drop_id_embed in zip(rich_ID, batch["drop_id_embeds"]):
                    if drop_id_embed == 1:
                        id_embeds_.append(torch.zeros_like(id_embed))
                    else:
                        id_embeds_.append(id_embed)
                id_embeds = torch.stack(id_embeds_)
                it_embeds_ = []
                for image_embed, drop_it_embed in zip(image_embeds, batch["drop_it_embeds"]):
                    if drop_it_embed == 1:
                        it_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        it_embeds_.append(image_embed)
                it_embeds= torch.stack(it_embeds_)
                # print(it_embeds.shape)
                noise_pred = anon_adapter(noisy_latents, mask, masked_images, timesteps,it_embeds,id_embeds,batch["normal_images"].to(accelerator.device, dtype=weight_dtype))
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                uloss,id_loss1,id_loss2 = compute_image_loss(noise_pred,noisy_latents,timesteps.cpu(),vae,noise_scheduler,x0=batch["images"],src_ID=src_poor_ID,id_model=ID_encoder)                
                loss += uloss
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                now = datetime.now()
                formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if accelerator.is_main_process and step % 10 == 0:
                    print("[{}]: Epoch {}, global_step {}, step {}, step_loss: {},idloss1:{},idloss2:{}".format(
                        formatted_time, epoch, global_step, step, avg_loss,id_loss1,id_loss2))  
                    # print("[{}]: Epoch {}, global_step {}, step {}, step_loss: {}".format(
                    #     formatted_time, epoch, global_step, step, avg_loss))            
            global_step += 1
            
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                print('---------------------aaaaaaaaaaaaaaaaaaaaaa-------------------------')
                # before saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
