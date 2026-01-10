import torch
from torch import nn
from torch.nn import functional as F
from config.config import *
from .mamba_core import ResidualMambaLayer


class VisionMambaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = MAMBA_PATCH_SIZE
        self.dim = MAMBA_DIM
        self.img_channels = IMG_CHANNELS
        self.attr_dim = ATTR_DIM
        self.patch_embedding = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.h_patches = IMG_SIZE // self.patch_size
        self.w_patches = IMG_SIZE // self.patch_size
        self.seq_len = self.h_patches * self.w_patches
        self.attr_embedding = nn.Linear(self.attr_dim, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + 1, self.dim))

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
        x_emb = self.patch_embedding(x)

        x_seq = x_emb.flatten(2).transpose(1, 2)

        c_emb = self.attr_embedding(cond).unsqueeze(1)

        x_input = torch.cat([c_emb, x_seq], dim=1)

        x_input = x_input + self.pos_embedding

        for layer in self.layers:
            x_input = layer(x_input)

        logits = self.output_head(x_input)

        return logits[:, :-1, :]

    def loss_function(self, pred_patches, real_imgs):
        B = real_imgs.shape[0]

        target_patches = real_imgs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        target_patches = target_patches.contiguous().view(B, self.img_channels, -1, self.patch_size, self.patch_size)
        target_patches = target_patches.permute(0, 2, 1, 3, 4)
        target_patches = target_patches.contiguous().view(B, self.seq_len, -1)

        loss = F.mse_loss(pred_patches, target_patches)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None):
        self.eval()

        if cond is None:
            cond = torch.randint(0, 2, (num_samples, self.attr_dim)).float().to(device)
        else:
            cond = cond.to(device)

        B = num_samples
        for layer in self.layers:
            layer.inference_start(batch_size=B)

        c_emb = self.attr_embedding(cond).unsqueeze(1)
        curr_input = c_emb + self.pos_embedding[:, 0:1, :]

        for layer in self.layers:
            curr_input = layer.inference_step(curr_input)

        next_patch_pred = self.output_head(curr_input)

        generated_patches = []
        generated_patches.append(next_patch_pred)
        for i in range(self.seq_len - 1):
            prev_patch_img = next_patch_pred.view(B, self.img_channels, self.patch_size, self.patch_size)
            patch_emb = self.patch_embedding(prev_patch_img)
            curr_input = patch_emb.view(B, 1, self.dim)
            curr_input = curr_input + self.pos_embedding[:, i + 1:i + 2, :]

            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)

            next_patch_pred = self.output_head(curr_input)
            generated_patches.append(next_patch_pred)

        full_seq = torch.cat(generated_patches, dim=1)
        full_seq = full_seq.view(B, self.h_patches, self.w_patches, self.img_channels, self.patch_size, self.patch_size)
        full_seq = full_seq.permute(0, 3, 1, 4, 2, 5)
        recon_img = full_seq.contiguous().view(B, self.img_channels, IMG_SIZE, IMG_SIZE)

        self.train()
        return recon_img
    
    def mamba_train_step(model, batch, device):
        images, attributes = batch
        images = images.to(device)
        attributes = attributes.to(device)
        pred_patches = model(images, attributes)
        loss = model.loss_function(pred_patches, images)
        metrics = {
            'mse': loss.item()
        }

        return loss, metrics
    
    def train_step_fn(self, batch, device):
        return self.mamba_train_step(batch, device)