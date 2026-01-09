from models.autoencoder.autoencoder import ConditionalVAE
from models.diffusion.diffusion_model import ConditionalDiffusion
from models.mamba.vision_mamba import VisionMambaModel
from config.config import MODEL_TYPE

def get_model_from_env():
    model_type = MODEL_TYPE
    
    print(f"Factory: Initializing model type '{model_type}'...")
    
    if model_type == 'vae':
        return ConditionalVAE(), "vae_celeba_experiment"
    elif model_type == 'diff':
         return ConditionalDiffusion(), "diff_celeba_experiment"
    elif model_type == 'mamba':
        return VisionMambaModel(), "mamba_celeba_experiment"
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Available: 'vae'")