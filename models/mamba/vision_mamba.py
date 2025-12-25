import torch
from torch import nn
from torch.nn import functional as F
from config.config import *
from .mamba_core import ResidualMambaLayer


class VisionMambaModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Parametri da Config
        self.patch_size = MAMBA_PATCH_SIZE
        self.dim = MAMBA_DIM
        self.img_channels = IMG_CHANNELS
        self.attr_dim = ATTR_DIM

        # 1. Patch Embedding (SOTA approach: Conv2d non-overlapping)
        # Trasforma ogni patch PxP in un vettore di dimensione D
        self.patch_embedding = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Calcolo lunghezza sequenza
        # Es: 64x64 img, patch 4 -> 16x16 patches -> 256 patches totali
        self.h_patches = IMG_SIZE // self.patch_size
        self.w_patches = IMG_SIZE // self.patch_size
        self.seq_len = self.h_patches * self.w_patches

        # 2. Embedding degli attributi (Conditioning)
        # Proiettiamo gli attributi one-hot nello stesso spazio dimensionale
        self.attr_embedding = nn.Linear(self.attr_dim, self.dim)

        # 3. Positional Embedding (Necessario per mantenere info spaziali)
        # +1 per il token degli attributi che aggiungeremo all'inizio
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + 1, self.dim))

        # 4. Core Mamba Layers (Logica identica al file originale)
        self.layers = nn.ModuleList()
        for i in range(MAMBA_LAYERS):
            self.layers.append(ResidualMambaLayer(
                dim=self.dim,
                state_size=MAMBA_STATE_SIZE,
                conv_kernel=MAMBA_CONV_KERNEL,
                expansion=MAMBA_EXPANSION
            ))

        # 5. Output Head (Prediction)
        # Predice i pixel della PROSSIMA patch.
        # Output dim = canali * patch_h * patch_w (es. 3 * 4 * 4 = 48 valori)
        self.pixels_per_patch = self.img_channels * self.patch_size * self.patch_size
        self.output_head = nn.Linear(self.dim, self.pixels_per_patch)

    def forward(self, x, cond):
        """
        x: [Batch, 3, 64, 64]
        cond: [Batch, 8]
        """
        B = x.shape[0]

        # A. Creazione sequenza patch
        # [B, 3, 64, 64] -> [B, Dim, 16, 16]
        x_emb = self.patch_embedding(x)

        # Flatten spaziale: [B, Dim, 16, 16] -> [B, Dim, 256] -> [B, 256, Dim]
        x_seq = x_emb.flatten(2).transpose(1, 2)

        # B. Integrazione Attributi (Conditioning)
        # Trattiamo gli attributi come un token speciale all'inizio (come [CLS] token in BERT/ViT)
        c_emb = self.attr_embedding(cond).unsqueeze(1)  # [B, 1, Dim]

        # Concatenazione: Sequenza diventa [AttrToken, Patch1, Patch2, ..., PatchN]
        # Nuova lunghezza: 257
        x_input = torch.cat([c_emb, x_seq], dim=1)

        # C. Aggiunta Positional Embedding
        x_input = x_input + self.pos_embedding

        # D. Passaggio nei layer Mamba
        for layer in self.layers:
            x_input = layer(x_input)

        # E. Output Projection
        # [B, L+1, Dim] -> [B, L+1, PixelsPerPatch]
        logits = self.output_head(x_input)

        # Rimuoviamo l'ultimo elemento perché stiamo predicendo il "prossimo"
        # Il token attributo predice patch 1, patch 1 predice patch 2, etc.
        # L'ultima patch predice "nulla" (fine sequenza), quindi prendiamo l'output fino a seq_len
        # Output shape: [B, 256, 48]
        return logits[:, :-1, :]

    def loss_function(self, pred_patches, real_imgs):
        """
        Calcola MSE Loss tra le patch predette e quelle reali.
        """
        B = real_imgs.shape[0]

        # Dobbiamo "patchificare" l'immagine target per confrontarla
        # Usiamo unfold per estrarre patch: [B, C, H, W] -> patches
        target_patches = real_imgs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # target_patches: [B, C, H_patches, W_patches, P, P]

        target_patches = target_patches.contiguous().view(B, self.img_channels, -1, self.patch_size, self.patch_size)
        target_patches = target_patches.permute(0, 2, 1, 3, 4)  # [B, SeqLen, C, P, P]
        target_patches = target_patches.contiguous().view(B, self.seq_len, -1)  # [B, SeqLen, PixelsPerPatch]

        loss = F.mse_loss(pred_patches, target_patches)

        # Metriche accessorie
        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None):
        """
        Generazione Autoregressiva (Patch by Patch).
        """
        self.eval()

        if cond is None:
            cond = torch.randint(0, 2, (num_samples, self.attr_dim)).float().to(device)
        else:
            cond = cond.to(device)

        B = num_samples # Questo è il batch size

        # 1. Inizializza lo stato dell'inferenza (Cache)
        for layer in self.layers:
            # MODIFICA FONDAMENTALE: Passo B (num_samples) a inference_start
            layer.inference_start(batch_size=B)

        # 2. Primo step: Processare il token attributo
        c_emb = self.attr_embedding(cond).unsqueeze(1)  # [B, 1, Dim]
        # Aggiungi pos embedding del token 0
        curr_input = c_emb + self.pos_embedding[:, 0:1, :]

        # Passaggio nei layer (inference_step)
        for layer in self.layers:
            curr_input = layer.inference_step(curr_input)

        # Predizione prima patch
        next_patch_pred = self.output_head(curr_input)  # [B, 1, Pixels]

        generated_patches = []
        generated_patches.append(next_patch_pred)

        # 3. Loop Autoregressivo per le restanti patch
        # Dobbiamo generare seq_len patch. La prima è generata dal cond.
        # Ne mancano seq_len - 1? No, ne generiamo seq_len totale.

        # Nota: Vision Mamba autoregressivo riusa l'output predetto come input successivo.
        # Dobbiamo proiettare i pixel predetti nello spazio embedding.
        # MA: Mamba si aspetta l'input dallo strato precedente, non i pixel grezzi rilanciati dentro la Conv2d.
        # Soluzione Generativa Standard:
        # Output Head -> Pixel -> (Conv2d o Linear Inverse) -> Embedding?
        # Per semplicità e stabilità in questo setup (che usa patch embedding conv):
        # Usiamo una proiezione lineare inversa ("Input Projection") invece della Conv2d durante la generazione
        # OPPURE (più robusto): Usiamo la Conv2d sui pixel generati.

        # Implementazione Loop:
        for i in range(self.seq_len - 1):
            # L'input allo step t è la patch generata allo step t-1
            # next_patch_pred shape: [B, 1, C*P*P]

            # Reshape a immagine per passare nella Conv2d (Patch Embedding)
            prev_patch_img = next_patch_pred.view(B, self.img_channels, self.patch_size, self.patch_size)

            # Embed
            patch_emb = self.patch_embedding(prev_patch_img)  # [B, Dim, 1, 1]
            curr_input = patch_emb.view(B, 1, self.dim)  # [B, 1, Dim]

            # Add Pos Embedding (index i+1 perché 0 era attributo)
            curr_input = curr_input + self.pos_embedding[:, i + 1:i + 2, :]

            # Mamba Step
            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)

            # Prediction
            next_patch_pred = self.output_head(curr_input)
            generated_patches.append(next_patch_pred)

        # 4. Ricostruzione Immagine dalle patch
        # generated_patches è una lista di [B, 1, Pixels]
        full_seq = torch.cat(generated_patches, dim=1)  # [B, SeqLen, Pixels]

        # Reshape a griglia
        # [B, H_patches, W_patches, C, P, P]
        full_seq = full_seq.view(B, self.h_patches, self.w_patches, self.img_channels, self.patch_size, self.patch_size)
        # Permute per mettere tutto in ordine [B, C, H_p, P, W_p, P]
        full_seq = full_seq.permute(0, 3, 1, 4, 2, 5)
        # Collapse finale
        recon_img = full_seq.contiguous().view(B, self.img_channels, IMG_SIZE, IMG_SIZE)

        self.train()
        return recon_img
    
    def mamba_train_step(model, batch, device):
        """
        Step di training specifico per Vision Mamba (Autoregressive).
        """
        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)

        # 1. Forward Pass
        # Il modello ritorna le patch predette
        pred_patches = model(images, attributes)

        # 2. Calcolo Loss
        # La funzione loss_function è definita dentro VisionMambaModel per incapsulare la logica di patchify del target
        loss = model.loss_function(pred_patches, images)
        
        # 3. Metriche per il logger
        metrics = {
            'mse': loss.item()
        }

        return loss, metrics
    
    def train_step_fn(self):
        """Restituisce la funzione di step di training specifica per il modello"""
        return self.mamba_train_step