import torch
import os
import time
import numpy as np
from .show_utils import save_images

def save_training_checkpoint(cpm, model, optimizer, epoch_count, force=False, interval=300):
    """
    Gestisce il salvataggio del checkpoint tramite CheckpointManager.
    Salva solo se è passato 'interval' tempo o se 'force' è True.
    """
    last_save = cpm.get_last_save_time()
    dt = time.time() - last_save
    
    if force or dt >= interval:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_count': epoch_count
        }
        cpm.save_checkpoint(state)
        print(' -> Checkpoint saved')


def save_snapshot(model, dataset, folder, epoch_count, device, num_samples=8):
    """
    Genera e salva un confronto visivo (Originale vs Generato).
    Adattata per gestire sia VAE (Ricostruzione) che Diffusion (Campionamento).
    """
    filename = f"snap_{epoch_count:04d}.png"
    filepath = os.path.join(folder, filename)

    # Mettiamo il modello in eval per disabilitare dropout/batchnorm durante la generazione
    model.eval()

    # 1. Selezione casuale indici dal dataset per avere un riferimento "Reale"
    indices = np.random.randint(0, len(dataset), num_samples)
    samples = [dataset[i] for i in indices]

    # Preparazione batch dati reali
    # real_imgs: [B, C, H, W] - Immagini vere
    # attrs: [B, 8] - Attributi corrispondenti (es. Maschio, Sorridente...)
    real_imgs = torch.stack([s[0] for s in samples]).to(device)
    attrs = torch.stack([s[1] for s in samples]).to(device)

    with torch.no_grad():
        # --- LOGICA ADATTIVA ---

        # CASO A: Il modello è un Generatore puro (es. Diffusion Model)
        # Verifichiamo se ha un metodo 'sample' esposto
        if hasattr(model, 'sample') and callable(model.sample):
            # Chiamiamo il sample passando gli attributi delle immagini reali.
            # Questo permette di confrontare:
            # "Reale (Uomo)" vs "Generato (Uomo)"
            # Nota: Non passiamo real_imgs, perché il diffusion parte dal rumore.
            recon_imgs = model.sample(num_samples=num_samples, device=device, cond=attrs)

        # CASO B: Il modello è un Autoencoder (es. VAE)
        # Funziona per ricostruzione diretta dell'input
        else:
            output = model(real_imgs, attrs)

            # Gestione output multipli (es. VAE ritorna (recon, mu, logvar))
            if isinstance(output, tuple):
                recon_imgs = output[0]
            else:
                recon_imgs = output

        # Salvataggio immagine:
        # Passiamo .cpu() perché matplotlib lavora su CPU/Numpy
        save_images(filepath, real_imgs.cpu(), recon_imgs.cpu(), figsize=(10, 5))

    print(f" -> Snapshot saved: {filepath}")

    # Rimettiamo il modello in training mode
    model.train()