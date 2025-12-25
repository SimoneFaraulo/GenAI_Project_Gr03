import os
import sys

# Importa i tuoi modelli qui
from models.autoencoder.autoencoder import ConditionalVAE
from models.diffusion.diffusion_model import ConditionalDiffusion
from models.mamba.vision_mamba import VisionMambaModel
from config.config import MODEL_TYPE
# from gan_model import ConditionalGAN (Esempio futuro)

def get_model_from_env():
    """
    Legge la variabile d'ambiente MODEL_TYPE e restituisce l'istanza del modello e il nome dell'esperimento.
    Default: 'vae'
    """
    model_type = MODEL_TYPE
    
    print(f"Factory: Initializing model type '{model_type}'...")
    
    match model_type:
        case 'vae':
            return ConditionalVAE(), "vae_celeba_experiment"
        case 'diff':
            return ConditionalDiffusion(), "diff_celeba_experiment"
        case 'mamba':
            return VisionMambaModel(), "mamba_celeba_experiment"
        case _:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Available: 'vae', 'diff', 'mamba'")