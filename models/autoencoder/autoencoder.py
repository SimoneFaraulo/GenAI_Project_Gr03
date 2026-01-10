import torch
from torch import nn
from torch.nn import functional as F
from config.config import *
from models.skip import Skip

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        hidden_dim = out_channels // 2

        self.residual = Skip(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        hidden_dim = out_channels // 2
        self.residual = Skip(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
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
        current_channels = in_channels + ATTR_EMBED_DIM
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(EncoderBlock(current_channels, h_dim))
            current_channels = h_dim

        self.encoder_net = nn.Sequential(*modules)

        num_downsamples = len(hidden_dims)
        self.final_spatial_dim = IMG_SIZE // (2 ** num_downsamples)
        
        if self.final_spatial_dim < 1:
            raise ValueError(f"Troppi layer ({num_downsamples}) per una immagine {IMG_SIZE}x{IMG_SIZE}. La risoluzione collassa a 0.")

        self.flatten_dim = hidden_dims[-1] * self.final_spatial_dim * self.final_spatial_dim
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, cond_emb):

        cond_expanded = cond_emb[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_expanded], dim=1)
        x = self.encoder_net(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc_mu(x), self.fc_var(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = HIDDEN_DIMS[::-1]
            
        self.initial_reshape_dim = hidden_dims[0]
        
        num_upsamples = len(hidden_dims)
        self.start_spatial_dim = IMG_SIZE // (2 ** num_upsamples)
        
        self.decoder_input = nn.Linear(
            latent_dim + ATTR_EMBED_DIM, 
            self.initial_reshape_dim * self.start_spatial_dim * self.start_spatial_dim
        )
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(DecoderBlock(hidden_dims[i], hidden_dims[i+1]))
            
        self.decoder_net = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z, cond_emb):
        z_cond = torch.cat([z, cond_emb], dim=1)

        x = self.decoder_input(z_cond)
        x = x.view(-1, self.initial_reshape_dim, self.start_spatial_dim, self.start_spatial_dim)
        
        x = self.decoder_net(x)
        x = self.final_layer(x)
        
        return x

class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()
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
        cond_emb = self.attr_embed(cond)
        
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
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)
            cond = (cond * 2) - 1
        else:
            cond = cond.to(device)

        cond_emb = self.attr_embed(cond)
        
        return self.decoder(z, cond_emb)
    
    def vae_train_step(self, batch, device):
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
        return self.vae_train_step(batch, device)