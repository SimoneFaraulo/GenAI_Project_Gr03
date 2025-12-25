import re
import torch
from torch import nn
from torch.nn import functional as F
from config.config import *
from models.skip import Skip

class EncoderBlock(nn.Module):
    """
    Blocco Encoder composto da:
    1. Downsampling (cambio dimensioni e canali)
    2. Residual Bottleneck (raffinamento con Skip connection)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1. Downsampling: Adatta l'input alle nuove dimensioni
        # Kernel 3x3 dispari, Stride 2 -> Dimezza H, W
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Dimensione bottleneck interna (es. 1/4 dei canali output)
        hidden_dim = out_channels // 2
        
        # 2. Residual Bottleneck (tramite classe Skip)
        # Struttura: 1x1 (reduce) -> 3x3 (process) -> 1x1 (expand)
        # Mantiene inalterate le dimensioni H, W, Canali
        self.residual = Skip(
            # Proiezione 1x1
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Convoluzione 3x3 (padding 1 preserva dimensioni)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Espansione 1x1
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.downsample(x)
        x = self.residual(x)
        x = self.final_activation(x)
        return x 

class DecoderBlock(nn.Module):
    """
    Blocco Decoder composto da:
    1. Upsampling (cambio dimensioni e canali)
    2. Residual Bottleneck (raffinamento con Skip connection)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1. Upsampling: Adatta l'input alle nuove dimensioni
        # Kernel 3x3 dispari, Stride 2, Output Padding 1 -> Raddoppia H, W esatti
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        hidden_dim = out_channels // 2
        
        # 2. Residual Bottleneck (tramite classe Skip)
        self.residual = Skip(
            # Proiezione 1x1
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout2d(0.1), # Dropout leggero solo nel decoder (nell'encoder potrebbe rovinare la qualità)
            
            # Convoluzione 3x3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Espansione 1x1
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.residual(x)
        x = self.final_activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = HIDDEN_DIMS
        
        # Input: RGB (3) + Attributi (3) = 6 canali
        # current_channels = in_channels + ATTR_DIM 
        
        # Poiché riceviamo l'embedding già fatto, concateniamo 64 canali extra.
        current_channels = in_channels + ATTR_EMBED_DIM
        
        modules = []
        # Costruzione blocchi Encoder
        # Es: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        for h_dim in hidden_dims:
            modules.append(EncoderBlock(current_channels, h_dim))
            current_channels = h_dim

        self.encoder_net = nn.Sequential(*modules)
        
        # Calcoliamo quante volte dimezziamo l'immagine
        num_downsamples = len(hidden_dims)
        # IMG_SIZE è importato da config (64). 
        # La dimensione spaziale finale sarà 64 / (2^num_layers)
        # Con 6 layer: 64 / 64 = 1. Con 4 layer: 64 / 16 = 4.
        self.final_spatial_dim = IMG_SIZE // (2 ** num_downsamples)
        
        if self.final_spatial_dim < 1:
            raise ValueError(f"Troppi layer ({num_downsamples}) per una immagine {IMG_SIZE}x{IMG_SIZE}. La risoluzione collassa a 0.")

        self.flatten_dim = hidden_dims[-1] * self.final_spatial_dim * self.final_spatial_dim
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, cond_emb):
        # x: [B, 3, 64, 64], cond: [B, 3]
        
        # Espansione attributi spaziale
        cond_expanded = cond_emb[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        #Es: [B, 3, 64, 64], [B, 3, 64, 64]
        x = torch.cat([x, cond_expanded], dim=1)
        # Es: [B, 6, 64, 64]
        x = self.encoder_net(x)
        # Es: [B, 6, 4, 4]
        x = torch.flatten(x, start_dim=1) # ignora la dimensione 0 del batch
        #Es : [B, 6*4*4]

        return self.fc_mu(x), self.fc_var(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = HIDDEN_DIMS[::-1] # [256, 128, 64, 32]
            
        self.initial_reshape_dim = hidden_dims[0] # 256
        
        num_upsamples = len(hidden_dims) # Numero totale di layer nel decoder
        # Nota: hidden_dims qui è già invertito, ma la lunghezza è la stessa dell'encoder
        # La dimensione spaziale iniziale deve essere simmetrica a quella finale dell'encoder
        self.start_spatial_dim = IMG_SIZE // (2 ** num_upsamples)
        
        # # Input lineare: Latent -> Tensore Appiattito
        # self.decoder_input = nn.Linear(
        #     latent_dim + ATTR_DIM, 
        #     self.initial_reshape_dim * self.start_spatial_dim * self.start_spatial_dim
        # )
        
        self.decoder_input = nn.Linear(
            latent_dim + ATTR_EMBED_DIM, 
            self.initial_reshape_dim * self.start_spatial_dim * self.start_spatial_dim
        )
        
        modules = []
        # Costruzione blocchi Decoder
        # Es: 4x4 -> 8x8 -> 16x16 -> 32x32
        for i in range(len(hidden_dims) - 1):
            modules.append(DecoderBlock(hidden_dims[i], hidden_dims[i+1]))
            
        self.decoder_net = nn.Sequential(*modules)
        
        # Blocco Finale per RGB (32x32 -> 64x64)
        self.final_layer = nn.Sequential(
            # Upsample finale 32 -> 64 (3x3 s2 p1 op1)
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv finale canali (32 -> 3)
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z, cond_emb):
        z_cond = torch.cat([z, cond_emb], dim=1) # [B, 131]

        x = self.decoder_input(z_cond) 
        # Reshape usando la dimensione calcolata dinamicamente
        x = x.view(-1, self.initial_reshape_dim, self.start_spatial_dim, self.start_spatial_dim)
        
        x = self.decoder_net(x) # -> [B, 32, 32, 32]
        x = self.final_layer(x) # -> [B, 3, 64, 64]
        
        return x

class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Definizione dell'Embedding unico per Encoder e Decoder
        self.attr_embed = nn.Sequential(
            nn.Linear(ATTR_DIM, ATTR_EMBED_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ATTR_EMBED_DIM, ATTR_EMBED_DIM),
        )
        
        self.encoder = Encoder(in_channels=IMG_CHANNELS, latent_dim=LATENT_DIM)
        self.decoder = Decoder(out_channels=IMG_CHANNELS, latent_dim=LATENT_DIM)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        # 3. Calcolo dell'embedding UNA SOLA VOLTA
        cond_emb = self.attr_embed(cond) # -> [Batch, ATTR_EMBED_DIM]
        
        mu, log_var = self.encoder(x, cond_emb)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z, cond_emb)
        return reconstruction, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, beta=BETA):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None):
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        if cond is None:
            # Genera attributi casuali (vettori -1/1)
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)
            cond = (cond * 2) - 1 # Se usi lo scaling -1/1
        else:
            cond = cond.to(device)
        
        # Anche nel sampling dobbiamo embeddare gli attributi prima di passarli al decoder
        cond_emb = self.attr_embed(cond)
        
        return self.decoder(z, cond_emb)
    
    def vae_train_step(model, batch, device):
        """
        Definisce come processare un batch per il VAE.
        Ritorna: (total_loss, dizionario_metriche_per_log)
        """
        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)
        
        # Forward
        recon, mu, log_var = model(images, attributes)
        
        # Loss
        loss, recon_loss, kl_loss = model.loss_function(recon, images, mu, log_var)
        
        # Metriche per il logger (normalizzate per batch size per leggibilità)
        bs = images.size(0)
        metrics = {
            'recon': recon_loss.item() / bs,
            'kl': kl_loss.item() / bs
        }
        
        return loss, metrics
    
    def train_step_fn(self):
        """Restituisce la funzione di step di training specifica per il modello"""
        return self.vae_train_step