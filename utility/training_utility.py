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

    # modello in eval per disabilitare dropout/batchnorm durante la generazione
    model.eval()

    # indici dal dataset casuali
    indices = np.random.randint(0, len(dataset), num_samples)
    samples = [dataset[i] for i in indices]

    real_imgs = torch.stack([s[0] for s in samples]).to(device)
    attrs = torch.stack([s[1] for s in samples]).to(device)

    with torch.no_grad():
        rows_to_save = [real_imgs.cpu()] # riga 1: immagini originali
        
        model_name = type(model).__name__.lower()
        
        if 'vae' in model_name:
            # Forward pass per ricostruire
            output = model(real_imgs, attrs)
            # Output (recon, mu, logvar)
            recon = output[0]
            
            rows_to_save.append(recon.cpu()) # riga 2: ricostruzione del VAE
        
        # Per tutti i modelli chiamiamo la generazione
        if hasattr(model, 'sample') and callable(model.sample):
            gen_imgs = model.sample(num_samples=num_samples, device=device, cond=attrs)
            rows_to_save.append(gen_imgs.cpu()) # riga 3 (o 2): generazione

        save_images(filepath, *rows_to_save, figsize=(10, 3 * len(rows_to_save)))

    print(f" -> Snapshot saved: {filepath}")

    # Rimettiamo il modello in training mode
    model.train()