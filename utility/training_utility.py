import torch
import os
import time
import numpy as np
from .show_utils import save_images

def save_training_checkpoint(cpm, model, optimizer, epoch_count, force=False, interval=300):
    """
    Gestisce il salvataggio del checkpoint tramite CheckpointManager.
    Salva solo se e' passato 'interval' tempo o se 'force' e' True.
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
    """
    filename = f"snap_{epoch_count:04d}.png"
    filepath = os.path.join(folder, filename)
    
    model.eval()
    
    # Selezione casuale indici
    indices = np.random.randint(0, len(dataset), num_samples)
    samples = [dataset[i] for i in indices]
    
    # Preparazione batch (immagini reali e attributi)
    real_imgs = torch.stack([s[0] for s in samples]).to(device)
    attrs = torch.stack([s[1] for s in samples]).to(device)
    
    with torch.no_grad():
        # CONTROLLO TIPO DI MODELLO
        # Se il nome della classe contiene "Diffusion", usiamo il metodo sample()
        if "Diffusion" in model.__class__.__name__:
            # Passiamo 'cond=attrs' per generare facce con gli stessi attributi delle originali
            recon_imgs = model.sample(num_samples=num_samples, device=device, cond=attrs)
        else:
            # Comportamento standard per VAE (forward)
            output = model(real_imgs, attrs)
            
            # Gestione output multipli (es. VAE ritorna (recon, mu, logvar))
            if isinstance(output, tuple):
                recon_imgs = output[0]
            else:
                recon_imgs = output
        
        # Salvataggio immagine confronto
        save_images(filepath, real_imgs.cpu(), recon_imgs.cpu(), figsize=(10, 5))
        
    print(f" -> Snapshot saved: {filepath}")
    model.train()