import torch
import sys
from config.config import DEVICE, checkpoint_base, EPOCHS, MODEL_TYPE, EXP_NUM, LEARNING_RATE
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


def diffusion_train_step(model, batch, device):
    """
    Step di training specifico per Diffusion Model.
    """
    images, attributes = batch
    images = images.to(device)
    attributes = attributes.to(device)

    # Il modello calcola tutto internamente (forward diffusion + loss)
    loss = model.compute_loss(images, attributes)

    # Metriche per il logger
    metrics = {
        'mse': loss.item()
    }

    return loss, metrics


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
    loss, metrics = model.loss_function(pred_patches, images)

    return loss, metrics


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
    checkpoint_folder = checkpoint_base(f"{exp_name}_{EXP_NUM}")
    cpm = CheckpointManager(checkpoint_folder, kept_checkpoints=3)
    
    # 4. Ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Selezione della step function corretta
    # Qui mappiamo il tipo di modello alla sua funzione di training
    current_model_type = MODEL_TYPE
    
    if current_model_type == 'vae':
        step_fn = vae_train_step
    elif current_model_type == 'diff':
        step_fn = diffusion_train_step
    elif current_model_type == 'mamba':
        step_fn = mamba_train_step
    else:
        raise ValueError(f"Unknown model type: {current_model_type}")

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