import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from .pipeline_flux_ipa import FluxPipeline
from .transformer_flux import FluxTransformer2DModel
from .attention_processor import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class Resampler(torch.nn.Module):
    def __init__(
        self,
        dim=4096,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=128,
        embedding_dim=1152,
        output_dim=4096,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = torch.nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList([
                    torch.nn.LayerNorm(dim),
                    torch.nn.MultiheadAttention(dim, heads, dim_head, batch_first=True),
                    torch.nn.LayerNorm(dim),
                    torch.nn.Sequential(
                        torch.nn.Linear(dim, dim * ff_mult),
                        torch.nn.GELU(),
                        torch.nn.Linear(dim * ff_mult, dim),
                    )
                ])
            )

    def forward(self, x):
        # x is [B, N, C]
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for norm1, attn, norm2, ff in self.layers:
            q = norm1(latents)
            kv = norm1(x)
            attn_out, _ = attn(q, kv, kv)
            latents = latents + attn_out
            latents = latents + ff(norm2(latents))
        
        return self.norm_out(self.proj_out(latents))

class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, use_resampler=False):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.use_resampler = use_resampler

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=torch.bfloat16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        if self.use_resampler:
            image_proj_model = Resampler(
                dim=4096,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=self.num_tokens,
                embedding_dim=1152,
                output_dim=self.pipe.transformer.config.joint_attention_dim,
                ff_mult=4,
            ).to(self.device, dtype=torch.bfloat16)
        else:
            image_proj_model = MLPProjModel(
                cross_attention_dim=self.pipe.transformer.config.joint_attention_dim, # 4096
                id_embeddings_dim=1152, 
                num_tokens=self.num_tokens,
            ).to(self.device, dtype=torch.bfloat16)
        
        return image_proj_model
    
    def set_ip_adapter(self):
        transformer = self.pipe.transformer
        ip_attn_procs = {} # 19+38=57
        for name in transformer.attn_processors.keys():
            if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = transformer.attn_processors[name]
    
        transformer.set_attn_processor(ip_attn_procs)
    
    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=self.image_encoder.dtype)
            
            if self.use_resampler:
                # Extract grid spatial features instead of pooler output
                encoder_outputs = self.image_encoder(clip_image, output_hidden_states=True)
                clip_image_embeds = encoder_outputs.hidden_states[-1]
            else:
                clip_image_embeds = self.image_encoder(clip_image).pooler_output
                
            clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.transformer.attn_processors.values():
            if isinstance(attn_processor, IPAFluxAttnProcessor2_0):
                attn_processor.scale = scale
    
    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=3.5,
        num_inference_steps=24,
        pipe=None,
        **kwargs,
    ):
        self.set_scale(scale)

        image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        p = pipe if pipe is not None else self.pipe
        images = p(
            prompt=prompt,
            joint_attention_kwargs={"image_emb": image_prompt_embeds},
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


if __name__ == '__main__':
    
    model_path = "black-forest-labs/FLUX.1-dev"
    image_encoder_path = "google/siglip-so400m-patch14-384"
    ipadapter_path = "./ip-adapter.bin"
        
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    pipe = FluxPipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.bfloat16
    )

    ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)
    
    image_dir = "./assets/images/2.jpg"
    image_name = image_dir.split("/")[-1]
    image = Image.open(image_dir).convert("RGB")
    image = resize_img(image)
    
    prompt = "a young girl"
    
    images = ip_model.generate(
        pil_image=image, 
        prompt=prompt,
        scale=0.7,
        width=960, height=1280,
        seed=42
    )

    images[0].save(f"results/{image_name}")