import torch
from torch import nn
import math
from config.config import *


class NoiseSchedule:
    """
    Gestisce lo schedule del rumore e pre-calcola tutti i coefficienti
    (alpha, beta, radici cumulative) necessari per il processo di diffusione forward e reverse.
    """
    def __init__(self, schedule_len=NOISE_SCHEDULE_L, s=0.008, device=DEVICE):
        """
        Inizializza lo schedule, calcolando i tensori per beta, alpha e le loro varianti cumulative su dispositivo specificato.
        
        Args:
            schedule_len (int): Lunghezza della sequenza di rumore (default da config).
            s (float): Shift parameter per lo schedule coseno (default 0.008).
            device (torch.device): Dispositivo su cui allocare i tensori.
        """
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
    """
    Implementa il 'Sinusoidal Time Embedding' per codificare l'informazione temporale (t)
    in vettori utilizzabili dalla rete neurale.
    """
    def __init__(self, dim, schedule_len=NOISE_SCHEDULE_L, device=DEVICE):
        """
        Pre-calcola la matrice degli embedding posizionali (seno e coseno) per tutti
        i passi temporali definiti dallo schedule.
        
        Args:
            dim (int): Dimensione finale del vettore di embedding.
            schedule_len (int): Numero totale di step temporali.
            device (torch.device): Dispositivo di esecuzione.
        """
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
        """
        Restituisce il vettore di embedding corrispondente al passo temporale t specificato.
        
        Args:
            t (torch.Tensor | int): Indice o tensore di indici temporali.

        Returns:
            torch.Tensor: Il vettore di embedding corrispondente.
        """
        return self.encoding[t]


class UNetBlock(nn.Module):
    """
    Rappresenta un blocco ricorsivo della U-Net. Ogni blocco contiene un encoder,
    un decoder e opzionalmente un blocco interno (inner block) per gestire diverse risoluzioni.
    """
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        """
        Configura i layer di convoluzione per l'encoder, il decoder e il combinatore finale,
        oltre a istanziare il blocco interno se necessario.
        
        Args:
            size (int): Risoluzione spaziale (H/W) dell'input a questo livello.
            outer_features (int): Numero di canali in input e output del blocco.
            inner_features (int): Numero di canali intermedi (per la trasformazione interna).
            cond_features (int): Dimensione del vettore di embedding delle condizioni.
            inner_block (nn.Module, optional): Il blocco UNet annidato (livello inferiore), se presente.
        """
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
        """
        Esegue il passaggio forward: codifica l'input, passa attraverso il blocco interno (se presente),
        concatena le feature (skip connection) e decodifica l'output.
        
        Args:
            x (torch.Tensor): Tensore di input [Batch, outer_features, size, size].
            time_encodings (torch.Tensor): Tensore degli embedding temporali.
            cond (torch.Tensor): Tensore degli embedding degli attributi.

        Returns:
            torch.Tensor: Output del blocco combinato con le skip connection.
        """
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
        """Metodo helper per costruire i blocchi sequenziali di layer convoluzionali e attivazioni. 
        
        Args:
            from_features (int): Canali in ingresso.
            to_features (int): Canali in uscita.
        """
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        """Metodo helper per costruire i blocchi sequenziali di layer convoluzionali e attivazioni. 
        
        Args:
            from_features (int): Canali in ingresso.
            to_features (int): Canali in uscita.
        """
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )

    def build_decoder(self, from_features, to_features):
        """Metodo helper per costruire i blocchi sequenziali di layer convoluzionali e attivazioni. 
        
        Args:
            from_features (int): Canali in ingresso.
            to_features (int): Canali in uscita.
        """
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )


class ConditionalDiffusion(nn.Module):
    """
    Classe principale del Modello DDPM.
    Gestisce la costruzione della U-Net, il calcolo della loss e il processo generazione.
    """
    def __init__(self):
        """
        Inizializza il modello, inclusi gli embedding degli attributi, lo schedule del rumore,
        il time encoding e la struttura U-Net ricorsiva.
        """
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
        """
        Propaga l'input attraverso la rete U-Net condizionata dal time-step t e dagli attributi (cond),
        restituendo il rumore predetto.
        
        Args:
            x (torch.Tensor): Batch di immagini rumorose [Batch, Channels, H, W].
            t (torch.Tensor): Batch di indici temporali [Batch].
            cond (torch.Tensor): Batch di vettori attributi (non encodati) [Batch, Attr_Dim].

        Returns:
            torch.Tensor: Il rumore predetto dalla rete.
        """
        enc = self.time_encoding[t]
        cond_emb = self.attr_embed(cond)
        x_in = self.pre(x)
        y = self.unet(x_in, enc, cond_emb)
        
        output = self.post(y)
        return output

    def build_unet(self, size, feat_list):
        """
        Costruisce ricorsivamente l'architettura U-Net basandosi sulla lista delle dimensioni delle feature fornite.
        
        Args:
            x (torch.Tensor): Batch di immagini rumorose [Batch, Channels, H, W].
            t (torch.Tensor): Batch di indici temporali [Batch].
            cond (torch.Tensor): Batch di vettori attributi (non encodati) [Batch, Attr_Dim].

        Returns:
            torch.Tensor: Il rumore predetto dalla rete.
        """
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:])
        else:
            inner_block = None

        return UNetBlock(size, feat_list[0], feat_list[1], ATTR_EMBED_DIM, inner_block)

    def compute_loss(self, x0, cond):
        """
        Calcola la loss MSE per il training. Campiona un istante t, aggiunge rumore all'immagine
        e calcola l'errore tra il rumore aggiunto e quello predetto dalla rete.
        
        Args:
            x0 (torch.Tensor): Batch di immagini originali (senza rumore).
            cond (torch.Tensor): Batch di attributi corrispondenti.

        Returns:
            torch.Tensor: Scalare rappresentante la MSE loss.
        """
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
    def sample(self, num_samples, device, cond=None, lam=LAMBDA, steps=NOISE_SCHEDULE_L, eta=1.0):
        """
        Esegue il processo di generazione usando l'algoritmo DDIM.
        
        Args:
            num_samples (int): Numero di immagini da generare.
            device (torch.device): Dispositivo di esecuzione.
            cond (torch.Tensor, optional): Attributi specifici. Se None, vengono generati random.
            lam (float): Coefficiente di "guidance" (CFG).
            steps (int): Numero di passi DDIM (default: NOISE_SCHEDULE_L (DDPM)).
            eta (float): Parametro di stocasticità (1.0 = DDPM (default), 0.0 = deterministico/DDIM).

        Returns:
            torch.Tensor: Batch di immagini generate [num_samples, Channels, H, W].
        """
        if cond is None:
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)

        cond0 = torch.zeros_like(cond).to(device)

        n = num_samples
        # Genera z_N ~ N(0, I)
        z = torch.randn(n, self.channels, self.img_size, self.img_size, device=device)
        was_training = self.training
        self.eval()

        # Selezione lineare dei passi temporali (sottoinsieme tau)
        # steps valori equidistanti tra 0 e schedule_len-1
        tau_seq = torch.linspace(0, self.noise_schedule.schedule_len - 1, steps, dtype=torch.long, device=device)

        # Loop inverso sui passi selezionati: i da steps-1 a 0
        for i in reversed(range(steps)):
            tau_curr = tau_seq[i]
            # Il passo precedente è l'indice successivo nella sequenza inversa (i-1), 
            # se siamo all'ultimo step (i=0) il precedente è "tempo -1" (che corrisponde a t<0).
            tau_prev = tau_seq[i-1] if i > 0 else -1

            # t deve essere un batch per il forward del modello
            t = tau_curr.view(1).expand(n)

            # Predizione del rumore con Classifier-Free Guidance
            g1 = self(z, t, cond)
            g0 = self(z, t, cond0)
            g = lam * g1 + (1 - lam) * g0

            # Calcolo del passo DDIM
            z = self.ddim_step(z, g, eta, tau_curr, tau_prev)

        if was_training:
            self.train()

        return z
    
    def ddim_step(self, zt, g, eta, tau_curr, tau_prev):
        """
        Esegue un singolo passo di aggiornamento DDIM.
        Calcola z_{tau_prev} partendo da z_{tau_curr} e dal rumore predetto g.
        """
        # Recupera alpha_curr
        a_curr = self.noise_schedule.alpha[tau_curr]
        
        # Recupera alpha_prev (gestendo il caso tau_prev < 0 -> alpha = 1.0)
        if tau_prev >= 0:
            a_prev = self.noise_schedule.alpha[tau_prev]
        else:
            a_prev = torch.tensor(1.0, device=zt.device)
        
        # Calcolo sigma
        sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_curr) * (1.0 - a_curr / a_prev))
        
        # Calcolo coefficienti c1 e c2
        c1 = torch.sqrt(a_prev / a_curr)
        c2 = torch.sqrt(1.0 - a_prev - sigma**2) - torch.sqrt(a_prev * (1.0 - a_curr) / a_curr)
        
        # Rumore casuale per il passo stocastico (se eta > 0)
        eps = torch.randn_like(zt)
        
        # Aggiornamento latente
        z_prev = c1 * zt + c2 * g + sigma * eps
        
        return z_prev
    
    def diffusion_train_step(self, batch, device):
        """
        Wrapper per il passo di training: sposta i batch sul device corretto, calcola la loss
        e restituisce le metriche per il logging.
        
        Args:
            batch (tuple): Una tupla (immagini, attributi) dal dataloader.
            device (torch.device): Dispositivo di esecuzione.

        Returns:
            tuple: (loss, metrics_dict) dove metrics_dict è un dizionario per il logging.
        """
        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)
        loss = self.compute_loss(images, attributes)

        metrics = {
            'mse': loss.item()
        }

        return loss, metrics
    
    def train_step_fn(self, batch, device):
        """
        Metodo di compatibilità per essere chiamato dal Trainer generico.
        
        Args:
            batch (tuple): Batch di dati (immagini, label).
            device (torch.device): Dispositivo corrente.
        """
        return self.diffusion_train_step(batch, device)