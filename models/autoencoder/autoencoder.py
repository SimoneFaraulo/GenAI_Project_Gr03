import torch
from torch import nn
from torch.nn import functional as F
from config.config import *
from models.skip import Skip

class EncoderBlock(nn.Module):
    """
    Blocco costruttivo dell'Encoder che combina un downsampling (convoluzione con stride)
    e una connessione residuale (Skip connection).
    """

    def __init__(self, in_channels, out_channels):
        """
        Inizializza il blocco di downsampling e il ramo residuale.

        Args:
            in_channels (int): Numero di canali in ingresso.
            out_channels (int): Numero di canali in uscita (dimensione delle feature map).
        """

        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), # dimezza H e W
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        hidden_dim = out_channels // 2

        self.residual = Skip( 
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, stride=1, padding=0), # shape inalterata
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),   # shape inalterata
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0), # shape inalterata
            nn.BatchNorm2d(out_channels),
        )
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        """
        Esegue il passaggio in avanti applicando downsampling, blocco residuo e attivazione finale.

        Args:
            x (torch.Tensor): Il tensore di input.

        Returns:
            torch.Tensor: Il tensore elaborato con dimensioni spaziali dimezzate.
        """

        x = self.downsample(x)
        x = self.residual(x)
        x = self.final_activation(x)

        return x  # [B, out_C, H/2, W/2]

class DecoderBlock(nn.Module):
    """
    Blocco costruttivo del Decoder che combina un upsampling (convoluzione trasposta)
    e una connessione residuale.
    """

    def __init__(self, in_channels, out_channels):
        """
        Inizializza il blocco di upsampling e il ramo residuale.

        Args:
            in_channels (int): Numero di canali in ingresso.
            out_channels (int): Numero di canali in uscita.
        """

        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # raddoppia W e H
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        hidden_dim = out_channels // 2
        self.residual = Skip(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, stride=1, padding=0),  # shape inalterata
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),  # shape inalterata
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0),  # shape inalterata
            nn.BatchNorm2d(out_channels)
        )
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Esegue il passaggio in avanti applicando upsampling, blocco residuo e attivazione.

        Args:
            x (torch.Tensor): Il tensore di input.

        Returns:
            torch.Tensor: Il tensore elaborato con dimensioni spaziali raddoppiate.
        """

        x = self.upsample(x)
        x = self.residual(x)
        x = self.final_activation(x)

        return x # [B, out_C, 2H, 2W]

class Encoder(nn.Module):
    """
    Rete Encoder principale.

    Comprime l'immagine di input (concatenata con l'embedding condizionale espanso)
    nello spazio latente, producendo i vettori media (mu) e log-varianza (log_var).
    """

    def __init__(self, in_channels, latent_dim, hidden_dims=None):
        """
        Costruisce dinamicamente i layer dell'encoder basandosi sulla lista delle dimensioni nascoste.

        Calcola inoltre la dimensione spaziale finale per appiattire correttamente il tensore prima dei layer lineari finali.

        Args:
            in_channels (int): Canali dell'immagine (es. 3 per RGB).
            latent_dim (int): Dimensione del vettore latente Z.
            hidden_dims (list, optional): Lista di interi che definisce la profondità dei layer.
        """

        super().__init__()
        if hidden_dims is None: hidden_dims = HIDDEN_DIMS # Default: [64, 128, 256, 512]
        current_channels = in_channels + ATTR_EMBED_DIM # all'inizio 3 + 128 = 131
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(EncoderBlock(current_channels, h_dim))
            current_channels = h_dim

        self.encoder_net = nn.Sequential(*modules)

        # 64 -> 32 -> 16 -> 8 -> 4
        num_downsamples = len(hidden_dims)
        self.final_spatial_dim = IMG_SIZE // (2 ** num_downsamples)
        
        if self.final_spatial_dim < 1:
            raise ValueError(f"Troppi layer ({num_downsamples}) per una immagine {IMG_SIZE}x{IMG_SIZE}. La risoluzione collassa a 0.")

        self.flatten_dim = hidden_dims[-1] * self.final_spatial_dim * self.final_spatial_dim # 512 * 4 * 4 = 8192
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, cond_emb):
        """
        Codifica l'input in parametri della distribuzione latente.

        L'embedding condizionale viene espanso spazialmente e concatenato all'immagine
        lungo la dimensione dei canali.

        Args:
            x (torch.Tensor): Immagine di input.
            cond_emb (torch.Tensor): Embedding degli attributi (vettore).

        Returns:
            tuple: Una coppia (mu, log_var) che rappresenta la distribuzione latente.
        """

        cond_expanded = cond_emb[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_expanded], dim=1)
        x = self.encoder_net(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc_mu(x), self.fc_var(x) # [B, latent_dim (512)]

class Decoder(nn.Module):
    """
    Rete Decoder principale.

    Ricostruisce l'immagine partendo dal vettore latente campionato (z)
    concatenato con l'embedding condizionale.
    """

    def __init__(self, out_channels, latent_dim, hidden_dims=None):
        """
        Inizializza il decoder costruendo i layer in ordine inverso rispetto all'encoder.

        Args:
            out_channels (int): Canali dell'immagine ricostruita (es. 3 per RGB).
            latent_dim (int): Dimensione del vettore latente Z.
            hidden_dims (list, optional): Lista dimensioni nascoste (invertita internamente).
        """

        super().__init__()
        if hidden_dims is None: hidden_dims = HIDDEN_DIMS[::-1] # inversa: [512, 256, 128, 64]
            
        self.initial_reshape_dim = hidden_dims[0] # 512
        
        num_upsamples = len(hidden_dims)
        self.start_spatial_dim = IMG_SIZE // (2 ** num_upsamples)
        
        self.decoder_input = nn.Linear(
            latent_dim + ATTR_EMBED_DIM, # 512 + 128 = 640
            self.initial_reshape_dim * self.start_spatial_dim * self.start_spatial_dim # 512 * 4 * 4 = 8192
        )
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(DecoderBlock(hidden_dims[i], hidden_dims[i+1])) # 512 -> 256 -> 128 -> 64 canali 
            
        self.decoder_net = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), # 64x32x32 -> 64x64x64
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1), # 3x64x64
            nn.Sigmoid() # per valori in [0, 1] perché le immagini sono normalizzate
        )

    def forward(self, z, cond_emb):
        """
        Decodifica il vettore latente in un'immagine.

        Concatena z e cond_emb, li proietta in una dimensione spaziale iniziale e applica i blocchi di upsampling.

        Args:
            z (torch.Tensor): Vettore latente campionato.
            cond_emb (torch.Tensor): Embedding degli attributi.

        Returns:
            torch.Tensor: L'immagine ricostruita (valori tra 0 e 1 tramite Sigmoid).
        """

        z_cond = torch.cat([z, cond_emb], dim=1)

        x = self.decoder_input(z_cond)
        x = x.view(-1, self.initial_reshape_dim, self.start_spatial_dim, self.start_spatial_dim) # [B, 8192] -> [B, 512, 4, 4]
        
        x = self.decoder_net(x)
        x = self.final_layer(x)
        
        return x # [B, 3, 64, 64]

class ConditionalVAE(nn.Module):
    """
    Modello completo Conditional Variational Autoencoder (CVAE).

    Coordina l'embedding degli attributi, l'encoding, la riparametrizzazione
    e il decoding.
    """

    def __init__(self):
        """
        Inizializza le sottoreti: l'embedder per gli attributi, l'Encoder e il Decoder.
        """

        super().__init__()
        self.attr_embed = nn.Sequential(
            nn.Linear(ATTR_DIM, ATTR_EMBED_DIM), # 3 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ATTR_EMBED_DIM, ATTR_EMBED_DIM),
        )
        
        self.encoder = Encoder(in_channels=IMG_CHANNELS, latent_dim=LATENT_DIM)
        self.decoder = Decoder(out_channels=IMG_CHANNELS, latent_dim=LATENT_DIM)

    def reparameterize(self, mu, log_var):
        """
        Applica il 'reparameterization trick' per permettere la backpropagation.

        Campiona z = mu + sigma * epsilon, dove epsilon è rumore normale standard.

        Args:
            mu (torch.Tensor): Media della distribuzione latente.
            log_var (torch.Tensor): Logaritmo della varianza della distribuzione.

        Returns:
            torch.Tensor: Il vettore latente z campionato.
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, cond):
        """
        Passaggio completo del modello durante il training/inferenza.

        Args:
            x (torch.Tensor): Immagine di input.
            cond (torch.Tensor): Vettore degli attributi grezzi.

        Returns:
            tuple: (immagine_ricostruita, mu, log_var)
        """

        cond_emb = self.attr_embed(cond)
        mu, log_var = self.encoder(x, cond_emb)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z, cond_emb)

        return reconstruction, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, beta=BETA):
        """
        Calcola la loss totale come somma della Loss di Ricostruzione (MSE) e della Divergenza di Kullback-Leibler (KL).

        Args:
            recon_x (torch.Tensor): Immagine ricostruita.
            x (torch.Tensor): Immagine originale target.
            mu (torch.Tensor): Media latente.
            log_var (torch.Tensor): Log-varianza latente.
            beta (float): Peso per la componente KL (regolarizzazione).

        Returns:
            tuple: (loss_totale, valore_recon_loss, valore_kl_loss)
        """
        # reduction='sum' somma l'errore su tutti i pixel e i sample del batch per bilanciare la scala rispetto alla KL anch'essa una somma
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        elbo_loss = recon_loss + beta * kl_loss

        return elbo_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None):
        """
        Genera nuove immagini campionando dallo spazio latente (rumore casuale)
        condizionato dagli attributi forniti o casuali.

        Args:
            num_samples (int): Numero di immagini da generare.
            device (torch.device): Dispositivo su cui allocare i tensori.
            cond (torch.Tensor, optional): Attributi specifici. Se None, vengono generati casualmente.

        Returns:
            torch.Tensor: Batch di immagini generate.
        """

        z = torch.randn(num_samples, LATENT_DIM).to(device)
        if cond is None:
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)
            cond = (cond * 2) - 1 # [0, 1] -> [-1, 1]
        else:
            cond = cond.to(device)

        cond_emb = self.attr_embed(cond)
        
        return self.decoder(z, cond_emb)
    
    def vae_train_step(self, batch, device):
        """
        Esegue un singolo step di training: forward pass e calcolo della loss.

        Args:
            batch (tuple): Coppia (immagini, attributi) dal DataLoader.
            device (torch.device): Dispositivo di calcolo.

        Returns:
            tuple: (loss_scalare, dizionario_metriche)
        """

        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)

        recon, mu, log_var = self(images, attributes)

        loss, recon_loss, kl_loss = self.loss_function(recon, images, mu, log_var)

        bs = images.size(0)
        metrics = {
            'recon': recon_loss.item() / bs,
            'kl': kl_loss.item() / bs
        }
        
        return loss, metrics
    
    def train_step_fn(self, batch, device):
        """
        Wrapper standardizzato per la funzione di training step, utile per
        interfacce di training generiche.
        """

        return self.vae_train_step(batch, device)