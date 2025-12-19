import torch
import sys
import os

from config.config import DEVICE, checkpoint_base, BATCH_SIZE, EPOCHS
from utility.checkpoint_manager import CheckpointManager
from data.data import data_loader, data_set
from utility.show_utils import parameter_count

# Nuovi moduli importati
from train.trainer import Trainer
from train.model_factory import get_model_from_env

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

# Puoi definire altre funzioni step qui, es: gan_train_step(...)


def main():
    # 1. Configurazione Epoche
    epochs = EPOCHS

    # 2. Istanziazione Modello Dinamica (da Env Var)
    try:
        model, exp_name = get_model_from_env()
    except ValueError as e:
        print(e)
        sys.exit(1)
        
    model = model.to(DEVICE)
    print(f"Parametri del modello: {parameter_count(model)}\n\n")
    
    # 3. Setup Cartelle e Checkpoint
    checkpoint_folder = checkpoint_base(exp_name)
    cpm = CheckpointManager(checkpoint_folder, kept_checkpoints=3)
    
    # 4. Ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # 5. Selezione della step function corretta
    # Qui mappiamo il tipo di modello alla sua funzione di training
    current_model_type = os.getenv('MODEL_TYPE', 'vae').lower()
    
    if current_model_type == 'vae':
        step_fn = vae_train_step
    else:
        # Fallback o logica per GAN
        step_fn = vae_train_step 

    # 6. Inizializzazione Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=data_loader,
        dataset=data_set,
        device=DEVICE,
        checkpoint_manager=cpm,
        train_step_fn=step_fn,
        checkpoint_folder=checkpoint_folder
    )
    
    # 7. Avvio
    trainer.load_checkpoint()
    trainer.train_loop(epochs=epochs, batches_per_epoch=len(data_loader))

if __name__ == '__main__':
    main()