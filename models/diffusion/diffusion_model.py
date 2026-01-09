import torch
from torch import nn
import math
from config.config import *


class NoiseSchedule:
    def __init__(self, schedule_len=NOISE_SCHEDULE_L, s=0.008, device=DEVICE):
        self.schedule_len = schedule_len
        t = torch.linspace(0.0, schedule_len, schedule_len + 1, device=device) / schedule_len
        a = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        a = a / a[0]
        self.beta = (1 - a[1:] / a[:-1]).clip(0.0, 0.99)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)
        self.one_minus_beta = 1 - self.beta
        self.one_minus_alpha = 1 - self.alpha
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.sqrt_1_alpha = torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta = torch.sqrt(self.one_minus_beta)


class TimeEncoding:
    def __init__(self, dim, schedule_len=NOISE_SCHEDULE_L, device=DEVICE):
        self.dim = dim
        self.schedule_len = schedule_len
        dim2 = dim // 2
        encoding = torch.zeros(schedule_len, dim, device=device)
        ang = torch.linspace(0.0, torch.pi / 2, schedule_len, device=device)
        logmul = torch.linspace(0.0, math.log(40), dim2, device=device)
        mul = torch.exp(logmul)

        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        self.encoding = encoding

    def __getitem__(self, t):
        return self.encoding[t]


class UNetBlock(nn.Module):
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        self.encoder = self.build_encoder(outer_features + cond_features, inner_features)
        self.decoder = self.build_decoder(inner_features + cond_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)
        self.inner = inner_block

    def forward(self, x, time_encodings, cond):
        x0 = x
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        x = torch.cat((x, cc), dim=1)
        y = self.encoder(x)

        if self.inner:
            y = self.inner(y, time_encodings, cond)

        half_size = self.size // 2
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size)
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half_size, half_size)
        y1 = torch.cat((y, cc, tt), dim=1)
        x1 = self.decoder(y1)
        x2 = torch.cat((x1, x0), dim=1)
        return self.combiner(x2)

    def build_combiner(self, from_features, to_features):
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )

    def build_decoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )


class ConditionalDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = IMG_SIZE
        self.channels = IMG_CHANNELS
        self.hidden_dims = DIFFUSION_HIDDEN_DIMS
        self.noise_schedule = NoiseSchedule(device=DEVICE)
        self.time_encoding = TimeEncoding(TIME_ENCODING_SIZE, device=DEVICE)
        self.attr_embed = nn.Sequential(
            nn.Linear(ATTR_DIM, ATTR_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(ATTR_EMBED_DIM, ATTR_EMBED_DIM),
        )
        self.pre = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_dims[0], 3, padding='same'),
            nn.ReLU()
        )
        self.unet = self.build_unet(self.img_size, self.hidden_dims)
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.hidden_dims[0], self.channels, 3, padding='same')
        )

    def forward(self, x, t, cond):
        enc = self.time_encoding[t]
        cond_emb = self.attr_embed(cond)
        x_in = self.pre(x)
        y = self.unet(x_in, enc, cond_emb)
        
        output = self.post(y)
        return output

    def build_unet(self, size, feat_list):
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:])
        else:
            inner_block = None

        return UNetBlock(size, feat_list[0], feat_list[1], ATTR_EMBED_DIM, inner_block)

    def compute_loss(self, x0, cond):
        batch_size = x0.shape[0]
        P = 0.2
        cond = cond.clone()
        u = torch.rand((batch_size,), device=x0.device)
        cond[u < P, :] = 0.0
        t = torch.randint(0, self.noise_schedule.schedule_len, (batch_size,), device=x0.device)
        eps = torch.randn_like(x0)
        sqrt_alpha = self.noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        sqrt_1_alpha = self.noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)
        zt = sqrt_alpha * x0 + sqrt_1_alpha * eps
        g = self(zt, t, cond)
        loss = nn.functional.mse_loss(g, eps)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None, lam=LAMBDA):
        if cond is None:
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)

        cond0 = torch.zeros_like(cond).to(device)

        n = num_samples
        z = torch.randn(n, self.channels, self.img_size, self.img_size, device=device)
        was_training = self.training
        self.eval()

        for kt in reversed(range(self.noise_schedule.schedule_len)):
            t = torch.tensor(kt, device=device).view(1).expand(n)
            beta = self.noise_schedule.beta[kt]
            sqrt_1_alpha = self.noise_schedule.sqrt_1_alpha[kt]
            sqrt_1_beta = self.noise_schedule.sqrt_1_beta[kt]
            sqrt_beta = self.noise_schedule.sqrt_beta[kt]

            g1 = self(z, t, cond)
            g0 = self(z, t, cond0)

            g = lam * g1 + (1 - lam) * g0

            mu = (z - beta / sqrt_1_alpha * g) / sqrt_1_beta

            if kt > 0:
                eps = torch.randn_like(z)
                z = mu + sqrt_beta * eps
            else:
                z = mu

        if was_training:
            self.train()

        return z
    
    def diffusion_train_step(self, batch, device):
        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)
        loss = self.compute_loss(images, attributes)

        metrics = {
            'mse': loss.item()
        }

        return loss, metrics
    
    def train_step_fn(self, batch, device):
        return self.diffusion_train_step(batch, device)