#
# !pip install torch torchvision einops tqdm wandb pickle5 numpy
import pickle
import wandb

# Original Imports
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

from time import time

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#
wandb.login(key="285708147e02543770d8ac9dde1c5d1c658d750f")  

#
# --- Configuration ---

# W&B Project Name
WANDB_PROJECT_NAME = "flamingo-ddpm"

# Dataset
DATASET_PATH = './data/flamingo.pkl' # <-- IMPORTANT: Point this to your pickle file

IMAGE_SIZE = 128         # Image resolution (must be a power of 2)
CHANNELS = 4             # Number of image channels

# Training
TRAIN_NUM_STEPS = 100000     # Total number of training steps
TRAIN_BATCH_SIZE = 32        # Batch size
LEARNING_RATE = 1e-4       # Learning rate
GRADIENT_ACCUMULATE_EVERY = 2 # Gradient accumulation steps

# Model
UNET_DIM = 64            # Base dimension for the U-Net
UNET_DIM_MULTS = (1, 2, 4, 8) # Multipliers for U-Net dimensions

# Diffusion
TIMESTEPS = 1000             # Number of diffusion timesteps

# Checkpointing and Logging
CHECKPOINTS_TO_SAVE = 10     # How many checkpoints to save during training
SAVE_EVERY = TRAIN_NUM_STEPS // CHECKPOINTS_TO_SAVE # Automatically calculate save interval
SAMPLE_EVERY = TRAIN_NUM_STEPS // CHECKPOINTS_TO_SAVE # Also sample images at each save
LOG_DIR = './results'        # Directory to save models and samples

# EMA (Exponential Moving Average)
EMA_DECAY = 0.995
STEP_START_EMA = 2000
UPDATE_EMA_EVERY = 10

# Multi-GPU (if available)
# List the IDs of the GPUs you want to use. E.g., [0] for one GPU, [0, 1] for two.
GPU_IDS = [0]

#
#   _   _      _
#  | | | | ___| |_ __   ___ _ __ ___
#  | |_| |/ _ \ | '_ \ / _ \ '__/ __|
#  |  _  |  __/ | |_) |  __/ |  \__ \
#  |_| |_|\___|_| .__/ \___|_|  |___/
#               |_|

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g

# Building Blocks
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


#
#   _   _            _                         _      _
#  | | | |_ __   ___| |_   _ __ ___   ___   __| | ___| |
#  | | | | '_ \ / _ \ __| | '_ ` _ \ / _ \ / _` |/ _ \ |
#  | |_| | | | |  __/ |_  | | | | | | (_) | (_| |  __/ |
#   \___/|_| |_|\___|\__| |_| |_| |_|\___/ \__,_|\___|_|
#
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


#    ____                     _                   _ _  __  __           _
#   / ___| __ _ _   _ ___ ___(_) __ _ _ __     __| (_)/ _|/ _|_   _ ___(_) ___  _ __
#  | |  _ / _` | | | / __/ __| |/ _` | '_ \   / _` | | |_| |_| | | / __| |/ _ \| '_ \
#  | |_| | (_| | |_| \__ \__ \ | (_| | | | | | (_| | |  _|  _| |_| \__ \ | (_) | | | |
#   \____|\__,_|\__,_|___/___/_|\__,_|_| |_|  \__,_|_|_| |_|  \__,_|___/_|\___/|_| |_|
#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)



#
#   ____        _                 _          _
#  |  _ \  __ _| |_ __ _ ___  ___| |_    ___| | __ _ ___ ___
#  | | | |/ _` | __/ _` / __|/ _ \ __|  / __| |/ _` / __/ __|
#  | |_| | (_| | || (_| \__ \  __/ |_  | (__| | (_| \__ \__ \
#  |____/ \__,_|\__\__,_|___/\___|\__|  \___|_|\__,_|___/___/
#

class FlamingoDataset(data.Dataset):
    def __init__(self, file_path, image_size):
        super().__init__()
        self.image_size = image_size
        
        print(f"Loading dataset from {file_path}...")
        with open(file_path, 'rb') as file:
            # Assuming the pickle file contains a tuple (labels, images)
            _, images = pickle.load(file)
        
        self.images = images
        # We will normalize to [-1, 1] after converting to tensor
        self.min_val = np.min(self.images)
        self.max_val = np.max(self.images)
        print(f"Dataset loaded. Original shape: {self.images.shape}. Min: {self.min_val}, Max: {self.max_val}")

        # --- NEW: Create a transform to resize the images ---
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts numpy array (H, W, C) or (C, H, W) to tensor (C, H, W) and scales to [0.0, 1.0]
            transforms.Resize(image_size, antialias=True),
            transforms.Lambda(lambda t: (t * 2) - 1) # Manually scale from [0, 1] to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        # Get image as numpy array, original shape is (C, H, W) -> (4, 100, 100)
        img_np = self.images[index].astype(np.float32)
    
        # --- FIX: Transpose from (C, H, W) to (H, W, C) for ToTensor ---
        # transforms.ToTensor() expects a numpy array in (Height, Width, Channels) format.
        img_np = img_np.transpose((1, 2, 0)) # New shape: (100, 100, 4)
        
        # Apply the transformations. 
        # ToTensor will convert it back to (C, H, W) tensor and scale to [0, 1].
        # Resize will then make it (C, 128, 128).
        img_tensor = self.transform(img_np)
        
        # Data Augmentation (on tensor)
        if torch.rand(1) < 0.5:
            img_tensor = transforms.functional.hflip(img_tensor)
        if torch.rand(1) < 0.5:
            img_tensor = transforms.functional.vflip(img_tensor)
    
        return img_tensor



#
#   _____          _                        _
#  |_   _| __ __ _(_)_ __   ___ _ __    ___| | __ _ ___ ___
#    | || '__/ _` | | '_ \ / _ \ '__|  / __| |/ _` / __/ __|
#    | || | | (_| | | | | |  __/ |    | (__| | (_| \__ \__ \
#    |_||_|  \__,_|_|_| |_|\___|_|     \___|_|\__,_|___/___/
#

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset, # <-- Takes the dataset object directly
        *,
        train_batch_size,
        train_lr,
        train_num_steps,
        gradient_accumulate_every,
        ema_decay,
        step_start_ema,
        update_ema_every,
        save_every,
        sample_every,
        logdir,
        rank = [0],
        num_workers = 0,
    ):
        super().__init__()
        self.model = torch.nn.DataParallel(diffusion_model, device_ids=rank)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_every = save_every
        self.sample_every = sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.logdir = Path(logdir)
        self.logdir.mkdir(exist_ok = True)
        
        # Use the passed dataset
        self.ds = dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.logdir / f'model-{milestone:06d}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.logdir / f'model-{milestone:06d}.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        t1 = time()
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).to(device=DEVICE)
                
                loss = self.model(data).sum()
                (loss / self.gradient_accumulate_every).backward()

            t0 = time()
            print(f'Step: {self.step} | Loss: {loss.item():.4f} | Time since last step: {t0 - t1:.3f}s')
            
            # --- W&B Logging ---
            wandb.log({
                'step': self.step,
                'loss': loss.item(),
                'time_per_step': t0 - t1,
            })
            t1 = time()
            
            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_every == 0:
                self.save(self.step)
                print(f"Checkpoint saved at step {self.step}")

            if self.step != 0 and self.step % self.sample_every == 0:
                milestone = self.step // self.sample_every
                batches = num_to_groups(16, self.batch_size) # Generate 16 sample images
                all_images_list = list(map(lambda n: self.ema_model.module.sample(batch_size=n), batches))
                
                all_images = torch.cat(all_images_list, dim=0)
                # Denormalize from [-1, 1] to [0, 1] for saving
                all_images = (all_images + 1) * 0.5
                
                # We have 4 channels, but save_image works best with 3 (RGB) or 1 (grayscale).
                # We'll save the first 3 channels as an example.
                utils.save_image(all_images[:, :3, :, :], str(self.logdir / f'sample-{self.step:06d}.png'), nrow = 4)
                print(f"Sample images saved at step {self.step}")
                
                # --- W&B Image Logging ---
                wandb.log({
                    "samples": [wandb.Image(img) for img in all_images[:, :3, :, :]]
                })

            self.step += 1

        print('Training complete.')



#
# --- Main Training ---

# 1. Initialize W&B
wandb.init(
    project=WANDB_PROJECT_NAME,
    config={
        "image_size": IMAGE_SIZE,
        "channels": CHANNELS,
        "unet_dim": UNET_DIM,
        "unet_dim_mults": UNET_DIM_MULTS,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_steps": TRAIN_NUM_STEPS,
    }
)

# 2. Instantiate the model and diffusion process
model = Unet(
    dim = UNET_DIM,
    dim_mults = UNET_DIM_MULTS,
    channels = CHANNELS
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = TIMESTEPS,
    channels = CHANNELS,
    loss_type = 'l1'
).to(DEVICE)

# 3. Create the dataset
dataset = FlamingoDataset(
    file_path=DATASET_PATH,
    image_size=IMAGE_SIZE
)

# 4. Instantiate the Trainer
trainer = Trainer(
    diffusion,
    dataset,
    train_batch_size = TRAIN_BATCH_SIZE,
    train_lr = LEARNING_RATE,
    train_num_steps = TRAIN_NUM_STEPS,
    gradient_accumulate_every = GRADIENT_ACCUMULATE_EVERY,
    ema_decay = EMA_DECAY,
    step_start_ema = STEP_START_EMA,
    update_ema_every = UPDATE_EMA_EVERY,
    save_every = SAVE_EVERY,
    sample_every = SAMPLE_EVERY,
    logdir = LOG_DIR,
    rank = GPU_IDS,
)

# 5. Start training
trainer.train()

# 6. Finish the W&B run
wandb.finish()



#
# --- Sampling from a Trained Model ---

# Path to your saved checkpoint

# model-090000
CHECKPOINT_PATH = f"{LOG_DIR}/model-{'090000'}.pt" # Path to the final checkpoint
NUM_SAMPLES = 16
SAMPLING_BATCH_SIZE = 4

# 1. Re-create the model and diffusion instances with the same parameters as training
model_sample = Unet(
    dim = UNET_DIM,
    dim_mults = UNET_DIM_MULTS,
    channels = CHANNELS
)

diffusion_sample = GaussianDiffusion(
    model_sample,
    image_size = IMAGE_SIZE,
    timesteps = TIMESTEPS,
    channels = CHANNELS,
    loss_type = 'l1'
).to(DEVICE)

# Wrap in DataParallel to match the saved state_dict keys
ema_model_sample = torch.nn.DataParallel(diffusion_sample, device_ids=GPU_IDS)

# 2. Load the checkpoint
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
data = torch.load(CHECKPOINT_PATH)
ema_model_sample.load_state_dict(data['ema'])

# 3. Sample images
print(f"Generating {NUM_SAMPLES} images...")
batches = num_to_groups(NUM_SAMPLES, SAMPLING_BATCH_SIZE)
all_images_list = list(map(lambda n: ema_model_sample.module.sample(batch_size=n), batches))
generated_images = torch.cat(all_images_list, dim=0)

# 4. Denormalize and save/display
generated_images = (generated_images + 1) * 0.5 # from [-1, 1] to [0, 1]

# Save the first 3 channels of the generated images
utils.save_image(generated_images[:,:3,:,:], str(Path(LOG_DIR) / 'final_samples.png'), nrow=4)
print("Samples saved to 'final_samples.png'")

# Optional: Display in notebook if you have matplotlib
try:
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    grid = make_grid(generated_images[:,:3,:,:], nrow=4).cpu()
    plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
except ImportError:
    print("Matplotlib not found. Cannot display images in notebook.")