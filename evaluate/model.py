import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletTransform3D(nn.Module):
    """
    Strict 3D Haar Wavelet Transform (like wdm-3d). 
    Perfect mathematical reconstruction with no overlapping artifacts.
    """
    def __init__(self):
        super().__init__()
        # Haar filters scaled by 1/sqrt(2) to preserve energy
        ll = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hh = torch.tensor([-1.0, 1.0]) / math.sqrt(2)
        
        # 3D outer products for the 8 subbands
        filters = []
        for fz in [ll, hh]:
            for fy in [ll, hh]:
                for fx in [ll, hh]:
                    filt = fz.view(2, 1, 1) * fy.view(1, 2, 1) * fx.view(1, 1, 2)
                    filters.append(filt)
                    
        # Shape: [8, 1, 2, 2, 2]
        self.register_buffer('weight_dwt', torch.stack(filters).unsqueeze(1))
        self.register_buffer('weight_idwt', torch.stack(filters).unsqueeze(1))
        
        self.register_buffer('channel_means', torch.zeros(8, 1, 1, 1))
        self.register_buffer('channel_stds', torch.ones(8, 1, 1, 1))

    def set_stats(self, means, stds):
        self.channel_means = means.view(8, 1, 1, 1)
        self.channel_stds = stds.view(8, 1, 1, 1)

    def dwt(self, x):
        """ Transforms [B, 1, 128, 128, 128] -> [B, 8, 64, 64, 64] and Normalizes """
        out = F.conv3d(x, self.weight_dwt, stride=2)
        out = (out - self.channel_means) / self.channel_stds
        return out

    def idwt(self, x):
        """ Un-normalizes and Transforms [B, 8, 64, 64, 64] -> [B, 1, 128, 128, 128] """
        x = (x * self.channel_stds) + self.channel_means
        return F.conv_transpose3d(x, self.weight_idwt, stride=2)

class GaussianDiffusion:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise):
        """ Forward process: Add Gaussian noise """
        alpha_cumprod = self.alphas_cumprod.to(x_0.device)[t].view(-1, 1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod) * x_0 + torch.sqrt(1.0 - alpha_cumprod) * noise

    def ddim_sample_loop(self, model, shape, start_t, noise, device, strides=50):
        """ Accelerated DDIM Reverse Process """
        b = shape[0]
        img = noise
        
        times = torch.linspace(start_t, 0, strides + 1).long().to(device)
        
        for i in range(strides):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                pred_noise = model(img, t_batch)
                
            alpha = self.alphas_cumprod.to(device)[t]
            alpha_next = self.alphas_cumprod.to(device)[t_next]
            
            pred_x0 = (img - torch.sqrt(1.0 - alpha) * pred_noise) / torch.sqrt(alpha)
            dir_xt = torch.sqrt(1.0 - alpha_next) * pred_noise
            img = torch.sqrt(alpha_next) * pred_x0 + dir_xt
            
        return img

# --- Basic 3D U-Net ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block3D(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, in_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.gn1(x))
        h = self.conv1(h)
        h = h + self.mlp(t_emb).view(-1, h.size(1), 1, 1, 1)
        h = self.act(self.gn2(h))
        return self.conv2(h) + (x if x.size(1) == h.size(1) else 0)

class UnconditionalWavUNet(nn.Module):
    """ Operating on the 8-channel Wavelet latent space """
    def __init__(self, in_channels=8, out_channels=8, base_dim=64):
        super().__init__()
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.init_conv = nn.Conv3d(in_channels, base_dim, 3, padding=1)
        
        # Down
        self.down1 = Block3D(base_dim, base_dim, time_dim)
        self.down2 = Block3D(base_dim, base_dim*2, time_dim)
        self.down3 = Block3D(base_dim*2, base_dim*4, time_dim)
        
        # Mid (Bottleneck at 16x16x16)
        self.mid1 = Block3D(base_dim*4, base_dim*4, time_dim)
        
        # Up
        self.up1 = Block3D(base_dim*8, base_dim*2, time_dim) # mid(4) + down3(4)
        self.up2 = Block3D(base_dim*4, base_dim, time_dim)   # up1(2) + down2(2)
        self.up3 = Block3D(base_dim*2, base_dim, time_dim)   # up2(1) + down1(1)
        self.up4 = Block3D(base_dim*2, base_dim, time_dim)   # up3(1) + x0(1)
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv3d(base_dim, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x0 = self.init_conv(x)
        
        # Downward Pass
        h1 = self.down1(x0, t_emb)
        
        h2_pool = F.avg_pool3d(h1, 2)
        h2 = self.down2(h2_pool, t_emb)
        
        h3_pool = F.avg_pool3d(h2, 2)
        h3 = self.down3(h3_pool, t_emb)
        
        # Bottleneck (3rd Pooling)
        h_mid_pool = F.avg_pool3d(h3, 2)
        h_mid = self.mid1(h_mid_pool, t_emb)
        
        # Upward Pass
        h_up = F.interpolate(h_mid, scale_factor=2, mode='nearest')
        h_up = self.up1(torch.cat([h_up, h3], dim=1), t_emb)
        
        h_up = F.interpolate(h_up, scale_factor=2, mode='nearest')
        h_up = self.up2(torch.cat([h_up, h2], dim=1), t_emb)
        
        h_up = F.interpolate(h_up, scale_factor=2, mode='nearest')
        h_up = self.up3(torch.cat([h_up, h1], dim=1), t_emb)
        
        # Final refinement at full resolution
        h_up = self.up4(torch.cat([h_up, x0], dim=1), t_emb)
        
        return self.out(h_up)