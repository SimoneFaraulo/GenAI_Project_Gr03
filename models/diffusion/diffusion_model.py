import torch
from torch import nn
import math
from config.config import *


class NoiseSchedule:
    def __init__(self, schedule_len=NOISE_SCHEDULE_L, s=0.008, device=DEVICE):
        self.schedule_len = schedule_len
        # Calcolo schedule coseno
        t = torch.linspace(0.0, schedule_len, schedule_len + 1, device=device) / schedule_len
        a = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        a = a / a[0]
        self.beta = (1 - a[1:] / a[:-1]).clip(0.0, 0.99)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_1_alpha = torch.sqrt(1 - self.alpha)


class TimeEncoding:
    def __init__(self, dim, schedule_len=NOISE_SCHEDULE_L, device=DEVICE):
        self.dim = dim
        self.schedule_len = schedule_len
        dim2 = dim // 2
        encoding = torch.zeros(schedule_len, dim, device=device)
        ang = torch.linspace(0.0, torch.pi / 2, schedule_len, device=device)
        logmul = torch.linspace(0.0, math.log(40), dim2, device=device)
        mul = torch.exp(logmul)

        # Vettorizzazione per efficienza (opzionale, ma raccomandata)
        # Manteniamo il ciclo se preferisci la leggibilità originale
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        self.encoding = encoding

    def __getitem__(self, t):
        return self.encoding[t]


class UNetBlock(nn.Module):
    # --- LOGICA ORIGINALE PRESERVATA (Vincolo 1) ---
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        # Nota: TIME_ENCODING_SIZE importato da config
        self.encoder = self.build_encoder(outer_features + cond_features, inner_features)
        self.decoder = self.build_decoder(inner_features + cond_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)
        self.inner = inner_block

    def forward(self, x, time_encodings, cond):
        x0 = x
        # Espansione condizionamento spaziale
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        x = torch.cat((x, cc), dim=1)
        y = self.encoder(x)

        if self.inner:
            y = self.inner(y, time_encodings, cond)

        # Parte Decoder
        half_size = self.size // 2
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size)
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half_size, half_size)
        y1 = torch.cat((y, cc, tt), dim=1)
        x1 = self.decoder(y1)

        # Skip connection
        x2 = torch.cat((x1, x0), dim=1)
        return self.combiner(x2)

    def build_combiner(self, from_features, to_features):
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            # Stride 2 dimezza la dimensione (es. 64 -> 32)
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )

    def build_decoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            # Transpose Stride 2 raddoppia (es. 32 -> 64)
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )


class ConditionalDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Parametri da Config
        self.img_size = IMG_SIZE
        self.channels = IMG_CHANNELS
        self.hidden_dims = DIFFUSION_HIDDEN_DIMS

        # Componenti ausiliarie
        self.noise_schedule = NoiseSchedule(device=DEVICE)
        self.time_encoding = TimeEncoding(TIME_ENCODING_SIZE, device=DEVICE)

        # Struttura Rete (Adattata a IMG_SIZE e CHANNELS)
        # Pre: da RGB (3) a Hidden[0] (64)
        self.pre = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_dims[0], 3, padding='same'),
            nn.ReLU()
        )

        # UNet Ricorsiva: Inizia da size=64
        self.unet = self.build_unet(self.img_size, self.hidden_dims)

        # Post: da Hidden[0] (64) a RGB (3)
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.hidden_dims[0], self.channels, 3, padding='same')
        )

    def forward(self, x, t, cond):
        """
        Predicts noise given x_t, t, and condition.
        """
        enc = self.time_encoding[t]
        x_in = self.pre(x)
        y = self.unet(x_in, enc, cond)
        output = self.post(y)
        return output

    # --- RICORSIONE ORIGINALE (Vincolo 1) ---
    def build_unet(self, size, feat_list):
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:])
        else:
            inner_block = None
        # size: dimensione corrente (es. 64)
        # feat_list[0]: outer (es. 64)
        # feat_list[1]: inner (es. 128)
        # ATTR_DIM: dimensione condizionamento (es. 3)
        return UNetBlock(size, feat_list[0], feat_list[1], ATTR_DIM, inner_block)

    # --- METODI AGGIUNTIVI PER COMPATIBILITÀ TRAINER ---

    def compute_loss(self, x, cond):
        """Calcola la loss MSE tra rumore predetto e rumore reale"""
        batch_size = x.shape[0]

        # 1. Campiona t uniformemente
        t = torch.randint(0, NOISE_SCHEDULE_L, (batch_size,), device=x.device)

        # 2. Genera rumore gaussiano
        eps = torch.randn_like(x)

        # 3. Ottieni coefficienti per t
        # Reshape per broadcasting corretto: [B, 1, 1, 1]
        sqrt_alpha_t = self.noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        sqrt_1_alpha_t = self.noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)

        # 4. Forward diffusion: z_t = sqrt_alpha * x + sqrt_1_alpha * eps
        z_t = sqrt_alpha_t * x + sqrt_1_alpha_t * eps

        # 5. Predizione del modello
        g = self(z_t, t, cond)

        # 6. Loss
        loss = nn.functional.mse_loss(g, eps)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None):
        """Genera immagini partendo da rumore puro (Reverse Diffusion)"""
        if cond is None:
            # Condizione random se non fornita (per test)
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)

        # Partiamo da rumore puro
        x = torch.randn(num_samples, self.channels, self.img_size, self.img_size, device=device)

        # Iterazione inversa da T-1 a 0
        for t_idx in range(NOISE_SCHEDULE_L - 1, -1, -1):
            t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)

            # Predizione rumore
            pred_noise = self(x, t_batch, cond)

            # Parametri per il passo inverso
            beta = self.noise_schedule.beta[t_idx]
            sqrt_1_alpha = self.noise_schedule.sqrt_1_alpha[t_idx]

            if t_idx > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            # Formula di update standard DDPM
            # x_{t-1} = 1/sqrt(1-beta) * (x_t - beta/sqrt(1-alpha) * pred_noise) + sigma * z
            coeff1 = 1 / torch.sqrt(1 - beta)
            coeff2 = beta / sqrt_1_alpha
            sigma = torch.sqrt(beta)

            x = coeff1 * (x - coeff2 * pred_noise) + sigma * z

        return x