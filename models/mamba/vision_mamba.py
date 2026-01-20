import torch
from torch import nn
from torch.nn import functional as F
from config.config import *
from .mamba_core import ResidualMambaLayer


class VisionMambaModel(nn.Module):
    """
    Modello VisionMamba per la generazione condizionale di immagini.
    Questa architettura tratta l'immagine come una sequenza di patch e utilizza
    blocchi Mamba (SSM) per modellare le dipendenze globali in modo autoregressivo,
    permettendo la generazione pixel-per-pixel (o patch-per-patch) guidata da attributi.
    """
    def __init__(self):
        """
        Inizializza i componenti del modello: embeddings per patch e attributi,
        embedding posizionale assoluto, stack di layer Mamba residui e testa di proiezione finale.
        I parametri dimensionali sono recuperati dal file di configurazione globale.
        """
        super().__init__()

        self.patch_size = MAMBA_PATCH_SIZE
        self.dim = MAMBA_DIM
        self.img_channels = IMG_CHANNELS
        self.attr_dim = ATTR_DIM
        self.patch_embedding = nn.Conv2d( # ogni patch viene mappato in un vettore di dimensione dim
            in_channels=self.img_channels,
            out_channels=self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.h_patches = IMG_SIZE // self.patch_size
        self.w_patches = IMG_SIZE // self.patch_size
        self.seq_len = self.h_patches * self.w_patches
        self.attr_embedding = nn.Linear(self.attr_dim, self.dim) # embedding per il condizionamento
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + 1, self.dim)) # +1 per il token di condizione

        self.layers = nn.ModuleList()
        for i in range(MAMBA_LAYERS):
            self.layers.append(ResidualMambaLayer(
                dim=self.dim,
                state_size=MAMBA_STATE_SIZE,
                conv_kernel=MAMBA_CONV_KERNEL,
                expansion=MAMBA_EXPANSION
            ))

        self.pixels_per_patch = self.img_channels * self.patch_size * self.patch_size
        self.output_head = nn.Linear(self.dim, self.pixels_per_patch)

    def forward(self, x, cond):
        """
        Esegue il passaggio forward per il training.
        Costruisce la sequenza di input concatenando l'embedding della condizione (attributi)
        con gli embedding dei patch dell'immagine, aggiunge le informazioni posizionali
        e processa il tutto attraverso i layer Mamba.

        Args:
            x (torch.Tensor): Batch di immagini reali (Batch, Channels, Height, Width).
            cond (torch.Tensor): Batch di vettori attributo (Batch, Attr_Dim).

        Returns:
            torch.Tensor: Logits predetti per il patch successivo (Batch, Seq_Len, Pixels_Per_Patch).
        """
        x_emb = self.patch_embedding(x) # (B, D, H_patch, W_patch)
        x_seq = x_emb.flatten(2).transpose(1, 2) # (B, L, D)

        c_emb = self.attr_embedding(cond).unsqueeze(1) # (B, ATTR_EMB) -> (B, D) -> (B, 1, D)

        x_input = torch.cat([c_emb, x_seq], dim=1)
        x_input = x_input + self.pos_embedding

        for layer in self.layers:
            x_input = layer(x_input)

        logits = self.output_head(x_input) # (B, L+1, D) -> (B, L+1, Pixels_Per_Patch)

        return logits[:, :-1, :] # l'ultimo elemento è superfluo (non ha un patch successivo da predire)

    def loss_function(self, pred_patches, real_imgs):
        """
        Calcola la Mean Squared Error (MSE) loss tra i patch predetti e quelli reali.
        La funzione si occupa di "srotolare" (unfold) le immagini originali in una sequenza
        di patch vettorializzati per renderle confrontabili con l'output del modello.

        Args:
            pred_patches (torch.Tensor): Tensore dei patch generati dal modello (B, L, Pixels_Per_Patch).
            real_imgs (torch.Tensor): Immagini target originali (B, C, H, W).

        Returns:
            torch.Tensor: Valore scalare della loss.
                """
        B = real_imgs.shape[0]

        # (B, C, H, W) -> (B, C, H_patches, W, patch_size) -> (B, C, H_patches, W_patches, patch_size, patch_size)
        target_patches = real_imgs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # (B, C, H_patches, W_patches, patch_size, patch_size) -> (B, C, L=H_patches*W_patches, patch_size, patch_size) 
        target_patches = target_patches.contiguous().view(B, self.img_channels, -1, self.patch_size, self.patch_size)
        target_patches = target_patches.permute(0, 2, 1, 3, 4) # (B, L, C, patch_size, patch_size)
        target_patches = target_patches.contiguous().view(B, self.seq_len, -1) # (B, L, Pixels_Per_Patch)

        loss = F.mse_loss(pred_patches, target_patches)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None, temperature=0.0):
        """
        Esegue la generazione autoregressiva (inferenza).
        Inizia processando il token di condizione, poi genera l'immagine patch dopo patch,
        utilizzando l'output predittivo di un passo come input per il passo successivo.
        Utilizza la cache dei layer Mamba per un'inferenza efficiente.

        Args:
            num_samples (int): Numero di immagini da generare.
            device (torch.device): Dispositivo su cui eseguire i calcoli.
            cond (torch.Tensor, optional): Attributi specifici per condizionare la generazione.
                                          Se None, vengono generati attributi casuali.
            temperature (float): Fattore di scala per il rumore gaussiano aggiunto alla predizione
                                 di ogni patch prima del passo successivo.
                                 Un valore > 0 introduce variabilità. Default: 0.0.

        Returns:
            torch.Tensor: Batch di immagini ricostruite (Batch, Channels, Height, Width).
        """
        self.eval()

        if cond is None:
            cond = torch.randint(0, 2, (num_samples, self.attr_dim)).float().to(device)
            cond = (cond * 2) - 1
        else:
            cond = cond.to(device)

        B = num_samples
        # inizializza la cache per l'inferenza in ogni layer Mamba
        for layer in self.layers:
            layer.inference_start(batch_size=B)

        c_emb = self.attr_embedding(cond).unsqueeze(1) # (B, ATTR_EMB) -> (B, D) -> (B, 1, D)
        curr_input = c_emb + self.pos_embedding[:, 0:1, :] # embedding posizionale per il token di condizione

        for layer in self.layers:
            curr_input = layer.inference_step(curr_input)

        next_patch_pred = self.output_head(curr_input) # primo patch predetto (B, 1, Pixels_Per_Patch)

        generated_patches = []
        generated_patches.append(next_patch_pred)
        
        # ciclo autoregressivo per generare i patch successivi
        for i in range(self.seq_len - 1):
            noise = torch.randn_like(next_patch_pred) * temperature # rumore per variabilità
            patch_input_for_next_step = next_patch_pred + noise
            
            # (B, 1, Pixels_Per_Patch) -> (B, C, patch_size, patch_size)
            prev_patch_img = patch_input_for_next_step.view(B, self.img_channels, self.patch_size, self.patch_size)
            patch_emb = self.patch_embedding(prev_patch_img) # (B, D, 1, 1)

            curr_input = patch_emb.view(B, 1, self.dim) # (B, 1, D)
            curr_input = curr_input + self.pos_embedding[:, i + 1:i + 2, :] # embedding posizionale i+1

            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)
            next_patch_pred = self.output_head(curr_input) # patch successivo predetto

            generated_patches.append(next_patch_pred)

        full_seq = torch.cat(generated_patches, dim=1) # (B, L, Pixels_Per_Patch)

        full_seq = full_seq.view(B, self.h_patches, self.w_patches, self.img_channels, self.patch_size, self.patch_size)
        full_seq = full_seq.permute(0, 3, 1, 4, 2, 5) # (B, C, H_patches, patch_size, W_patches, patch_size)
        recon_img = full_seq.contiguous().view(B, self.img_channels, IMG_SIZE, IMG_SIZE) # (B, C, H, W)

        self.train()
        return recon_img
    
    def mamba_train_step(self, batch, device):
        """
        Metodo di utilità statica per eseguire un singolo passo di training.
        Gestisce il passaggio dei dati al dispositivo, il forward pass e il calcolo della loss.

        Args:
            batch (tuple): Una tupla contenente (immagini, attributi).
            device (torch.device): Dispositivo di calcolo.

        Returns:
            tuple: Una tupla contenente (loss, dizionario delle metriche).
        """
        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)
        pred_patches = self(images, attributes)
        loss = self.loss_function(pred_patches, images)
        metrics = {
            'mse': loss.item()
        }

        return loss, metrics
    
    def train_step_fn(self, batch, device):
        """
        Wrapper di istanza che richiama la logica di training definita in mamba_train_step.

        Args:
            batch (tuple): Batch di dati corrente.
            device (torch.device): Dispositivo di calcolo.

        Returns:
            tuple: Risultato dello step di training (loss, metrics).
        """
        return self.mamba_train_step(batch, device)