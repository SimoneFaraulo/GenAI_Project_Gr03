import os

DEVICE = os.getenv("TORCH_DEVICE", "cuda")
HOME_DIRECTORY = os.getenv("HOME", ".")

CHECKPOINT_DIRECTORY = os.getenv("CHECKPOINT_DIRECTORY", None)
if not CHECKPOINT_DIRECTORY:
    CHECKPOINT_DIRECTORY = os.path.join(HOME_DIRECTORY, "temp", "CHECKPOINTS")
CKP_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", '300'))
FORCE_CKP = int(os.getenv("FORCE_CKP", '0'))

PROGRESS_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", '30'))

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./dataset")

MODEL_TYPE = os.getenv('MODEL_TYPE', 'diff').lower()
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0005"))
EXP_NUM = os.getenv("EXP_NUM", "1")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))

# Comuni a tutti
IMG_CHANNELS = 3          # RGB
IMG_SIZE = 64             # Dimensione richiesta (CelebA)
ATTR_DIM = 3              # 3 per -1 e 1 # 8 per One - Hot
COND_SHAPE = (8,)


ATTR_EMBED_DIM = int(os.getenv("ATTR_EMBED_DIM", "128"))       # Nuova dimensione dopo l'embedding

# Funzione helper per parsare liste di interi da stringhe env (es: "32,64,128")
def parse_int_list(env_var_name, default_list):
    val = os.getenv(env_var_name)
    if val:
        try:
            # Divide per virgola e converte ogni elemento in int
            return [int(x.strip()) for x in val.split(',')]
        except ValueError:
            print(f"Attenzione: Formato non valido per {env_var_name}. Uso il default.")
            return default_list
    return default_list

# Params del VAE
HIDDEN_DIMS = parse_int_list("HIDDEN_DIMS", [32, 64, 128, 256]) # 
LATENT_DIM = int(os.getenv("LATENT_DIM", "128")) # Dimensione spazio latente
BETA = float(os.getenv("BETA", "1.0"))           # Peso della KL Loss

# Params del Diffusion
TIME_ENCODING_SIZE = int(os.getenv("TIME_ENCODING_SIZE", "64"))
NOISE_SCHEDULE_L = int(os.getenv("NOISE_SCHEDULE_L", "1000"))
DIFFUSION_HIDDEN_DIMS = parse_int_list("DIFFUSION_HIDDEN_DIMS", [64, 128, 256, 512, 1024])
LAMBDA = float(os.getenv("LAMBDA", "3.0"))

# Params Mamba (Vision Mamba)
MAMBA_PATCH_SIZE = 4            # Dimensione della patch (SOTA per 64x64 è 4 o 8)
MAMBA_DIM = 512                 # Embedding dimension (D)
MAMBA_STATE_SIZE = 16           # State size (N)
MAMBA_EXPANSION = 2             # Expansion factor (E)
MAMBA_LAYERS = 8                # Numero di layer
MAMBA_CONV_KERNEL = 4           # Kernel della conv locale 1D interna a Mamba

## TODO

def checkpoint_base(name):
    return os.path.join(CHECKPOINT_DIRECTORY, name)