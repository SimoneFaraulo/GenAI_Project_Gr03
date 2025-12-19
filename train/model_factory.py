import os
import sys

# Importa i tuoi modelli qui
from models.autoencoder.autoencoder import ConditionalVAE
# from gan_model import ConditionalGAN (Esempio futuro)

def get_model_from_env():
    """
    Legge la variabile d'ambiente MODEL_TYPE e restituisce l'istanza del modello.
    Default: 'vae'
    """
    model_type = os.getenv('MODEL_TYPE', 'vae').lower()
    
    print(f"Factory: Initializing model type '{model_type}'...")
    
    if model_type == 'vae':
        return ConditionalVAE(), "vae_celeba_experiment"
    
    # Esempio per estensione futura:
    # elif model_type == 'gan':
    #     return ConditionalGAN(), "gan_celeba_experiment"
    
    # elif model_type == 'vit':
    #     return VisionTransformer(), "vit_celeba_experiment"
    
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Available: 'vae'")