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
        # Parametri da Config
        self.img_size = IMG_SIZE
        self.channels = IMG_CHANNELS
        self.hidden_dims = DIFFUSION_HIDDEN_DIMS

        # Componenti ausiliarie
        self.noise_schedule = NoiseSchedule(device=DEVICE)
        self.time_encoding = TimeEncoding(TIME_ENCODING_SIZE, device=DEVICE)
        
        # --- NUOVO: Attribute Embedding ---
        # Proietta il vettore attributi (dim 3) in uno spazio più ampio (dim 64)
        self.attr_embed = nn.Sequential(
            nn.Linear(ATTR_DIM, ATTR_EMBED_DIM),
            nn.SiLU(), # SiLU è standard nei Diffusion Models
            nn.Linear(ATTR_EMBED_DIM, ATTR_EMBED_DIM),
        )
        # ----------------------------------

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
        
        # Applicazione embedding
        cond_emb = self.attr_embed(cond)
        
        x_in = self.pre(x)
        y = self.unet(x_in, enc, cond_emb)
        
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
        #return UNetBlock(size, feat_list[0], feat_list[1], ATTR_DIM, inner_block)
        
        # MODIFICA: Passiamo ATTR_EMBED_DIM invece di ATTR_DIM
        return UNetBlock(size, feat_list[0], feat_list[1], ATTR_EMBED_DIM, inner_block)

    # --- METODI AGGIUNTIVI PER COMPATIBILITÀ TRAINER ---
    def compute_loss(self, x0, cond):
        """
        Calcola la loss replicando esattamente la logica di training con
        Classifier-Free Guidance (dropout del condizionamento).
        """
        batch_size = x0.shape[0]

        # Logica "MIA IMPLEMENTAZIONE": Remove conditioning with probability P=0.2
        P = 0.2
        # Creiamo una copia per non modificare il tensore originale nel batch
        cond = cond.clone()
        u = torch.rand((batch_size,), device=x0.device)
        
        # Azzeriamo il condizionamento dove u < P
        # Questa riga funziona perfettamente anche con i vettori -1/1.
        # Sostituisce i valori reali (-1 o 1) con 0.0.
        cond[u < P, :] = 0.0

        # 2. Scelta casuale dei timestep (uno per ogni sample nel minibatch)
        t = torch.randint(0, self.noise_schedule.schedule_len, (batch_size,), device=x0.device)

        # 3. Generazione del rumore casuale (eps)
        eps = torch.randn_like(x0)

        # 4. Calcolo dell'immagine latente (zt)
        # Recuperiamo i coefficienti dallo schedule e facciamo reshape per il broadcasting [B, 1, 1, 1]
        sqrt_alpha = self.noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        sqrt_1_alpha = self.noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)

        # Formula esatta: zt = sqrt_alpha * x + sqrt_1_alpha * eps
        zt = sqrt_alpha * x0 + sqrt_1_alpha * eps

        # 5. Output della rete (stima di eps)
        # g = model(zt, t, cond)
        g = self(zt, t, cond)

        # 6. Calcolo Loss (MSE tra rumore predetto e rumore reale)
        loss = nn.functional.mse_loss(g, eps)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None, lam=LAMBDA):
        """
        Genera immagini usando il sampling DDPM con Classifier-Free Guidance.
        Basato sul codice 'FUNZIONANTE MIO' dell'utente.

        Args:
            num_samples: numero di immagini da generare
            device: device su cui eseguire
            cond: tensore degli attributi (se None, viene generato casualmente)
            lam: scala della guida (guidance scale).
                 lam=1.0 -> sampling condizionato standard.
                 lam>1.0 -> forza maggiormente gli attributi.
        """
        # 1. Setup Condizioni
        if cond is None:
            # Genera attributi random se non forniti
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)

        # cond0 serve per la guida "senza etichetta" (tutto zeri come nel tuo codice)
        cond0 = torch.zeros_like(cond).to(device)

        # 2. Setup Iniziale
        n = num_samples
        # Usa self.img_size e self.channels definiti nella classe
        z = torch.randn(n, self.channels, self.img_size, self.img_size, device=device)

        # 3. Imposta Eval Mode (CRUCIALE come hai notato)
        # Salviamo lo stato precedente per ripristinarlo alla fine
        was_training = self.training
        self.eval()

        # 4. Loop di Reverse Diffusion
        for kt in reversed(range(self.noise_schedule.schedule_len)):
            # Preparazione batch temporale
            t = torch.tensor(kt, device=device).view(1).expand(n)

            # Recupero parametri dallo schedule (usando self.noise_schedule)
            beta = self.noise_schedule.beta[kt]
            sqrt_1_alpha = self.noise_schedule.sqrt_1_alpha[kt]  # sqrt(1 - alpha_cumprod)
            sqrt_1_beta = self.noise_schedule.sqrt_1_beta[kt]  # sqrt(1 - beta) aka sqrt(alpha)
            sqrt_beta = self.noise_schedule.sqrt_beta[kt]  # sigma

            # Stima dell'errore (noise prediction) con CFG
            # Nota: self(x, t, c) chiama il forward della classe
            g1 = self(z, t, cond)  # Predizione condizionata
            g0 = self(z, t, cond0)  # Predizione incondizionata

            # Combinazione (Guidance)
            g = lam * g1 + (1 - lam) * g0

            # Calcolo della media (mu)
            # Formula: mu = 1/sqrt(alpha) * (x - beta/sqrt(1-alpha_cumprod) * eps)
            # Nel tuo codice: sqrt_1_beta corrisponde a sqrt(alpha_t)
            mu = (z - beta / sqrt_1_alpha * g) / sqrt_1_beta

            # Generazione e aggiunta del rumore
            if kt > 0:
                eps = torch.randn_like(z)
                z = mu + sqrt_beta * eps
            else:
                z = mu

        if was_training:
            self.train()

        return z