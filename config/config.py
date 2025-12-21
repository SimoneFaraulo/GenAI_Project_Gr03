import os

DEVICE = os.getenv("TORCH_DEVICE", "cuda")
HOME_DIRECTORY = os.getenv("HOME", ".")

CHECKPOINT_DIRECTORY = os.getenv("CHECKPOINT_DIRECTORY", None)
if not CHECKPOINT_DIRECTORY:
    CHECKPOINT_DIRECTORY = os.path.join(HOME_DIRECTORY, "temp", "CHECKPOINTS")

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./dataset")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
MODEL_TYPE = os.getenv('MODEL_TYPE', 'vae').lower()
EXP_NUM = os.getenv("EXP_NUM", "1")
CKP_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", '300'))
FORCE_CKP = int(os.getenv("FORCE_CKP", '0'))
PROGRESS_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", '30'))

# Comuni a tutti
IMG_CHANNELS = 3          # RGB
IMG_SIZE = 64             # Dimensione richiesta (CelebA)
ATTR_DIM = 8              # One - Hot
COND_SHAPE = (8,)

# Params del VAE
HIDDEN_DIMS = [32, 64, 128, 256]                 # Canali progressivi
LATENT_DIM = int(os.getenv("LATENT_DIM", "128")) # Dimensione spazio latente

# Params del Diffusion
TIME_ENCODING_SIZE = int(os.getenv("TIME_ENCODING_SIZE", "64"))
NOISE_SCHEDULE_L = int(os.getenv("NOISE_SCHEDULE_L", "1000"))
DIFFUSION_HIDDEN_DIMS = [64, 128, 256, 512] # Più profondo per 64x64
LAMBDA = float(os.getenv("LAMBDA", "3.0"))

## TODO

def checkpoint_base(name):
    return os.path.join(CHECKPOINT_DIRECTORY, name)
