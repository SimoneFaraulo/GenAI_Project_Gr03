"""
Modulo di configurazione globale per il progetto.

Questo script gestisce il caricamento delle variabili d'ambiente necessarie per
l'inizializzazione del dispositivo, la gestione dei percorsi (dataset, pesi, checkpoint)
e la definizione degli iperparametri per i modelli (Vae, Diffusion, Mamba).
Fornisce valori di default per garantire il funzionamento anche in assenza di
configurazioni specifiche nell'ambiente.
"""


import os

DEVICE = os.getenv("TORCH_DEVICE", "cuda")
HOME_DIRECTORY = os.getenv("HOME", ".")
WEIGHTS_DIRECTORY = os.path.join(HOME_DIRECTORY, "weights")
CHECKPOINT_DIRECTORY = os.getenv("CHECKPOINT_DIRECTORY", None)
if not CHECKPOINT_DIRECTORY:
    CHECKPOINT_DIRECTORY = os.path.join(HOME_DIRECTORY, "temp", "CHECKPOINTS")
CKP_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", '300'))
FORCE_CKP = int(os.getenv("FORCE_CKP", '0'))

PROGRESS_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", '30'))

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./dataset")

MODEL_TYPE = os.getenv('MODEL_TYPE', 'mamba').lower()
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0005"))
EXP_NUM = os.getenv("EXP_NUM", "1")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

IMG_CHANNELS = 3
IMG_SIZE = 64
ATTR_DIM = 3
COND_SHAPE = (8,)


ATTR_EMBED_DIM = int(os.getenv("ATTR_EMBED_DIM", "128"))

def parse_int_list(env_var_name, default_list):
    """
    Recupera una variabile d'ambiente e converte la sua stringa separata da virgole
    in una lista di numeri interi.

    Gestisce eventuali errori di formattazione restituendo la lista di default fornita.

    Args:
        env_var_name (str): Il nome della variabile d'ambiente da leggere.
        default_list (list): La lista di interi da utilizzare come fallback se la variabile è vuota, assente o malformata.
    """

    val = os.getenv(env_var_name)
    if val:
        try:
            return [int(x.strip()) for x in val.split(',')]
        except ValueError:
            print(f"Attenzione: Formato non valido per {env_var_name}. Uso il default.")
            return default_list
    return default_list

HIDDEN_DIMS = parse_int_list("HIDDEN_DIMS", [64, 128, 256, 512])
LATENT_DIM = int(os.getenv("LATENT_DIM", "512"))
BETA = float(os.getenv("BETA", "0.5"))

TIME_ENCODING_SIZE = int(os.getenv("TIME_ENCODING_SIZE", "256"))
NOISE_SCHEDULE_L = int(os.getenv("NOISE_SCHEDULE_L", "1000"))
DIFFUSION_HIDDEN_DIMS = parse_int_list("DIFFUSION_HIDDEN_DIMS", [64, 128, 256, 512, 1024])
LAMBDA = float(os.getenv("LAMBDA", "3.0"))

MAMBA_PATCH_SIZE = int(os.getenv("MAMBA_PATCH_SIZE", "8"))
MAMBA_DIM = int(os.getenv("MAMBA_DIM", "512"))
MAMBA_STATE_SIZE = int(os.getenv("MAMBA_STATE_SIZE", "16"))
MAMBA_EXPANSION = int(os.getenv("MAMBA_EXPANSION", "2"))
MAMBA_LAYERS = int(os.getenv("MAMBA_LAYERS", "8"))
MAMBA_CONV_KERNEL = int(os.getenv("MAMBA_CONV_KERNEL", "4"))

def checkpoint_base(name):
    """
    Costruisce il percorso completo per un file di checkpoint combinando la
    directory base configurata con il nome specificato.

    Args:
        name (str): Il nome del file o della sottocartella del checkpoint da raggiungere.
    """

    return os.path.join(CHECKPOINT_DIRECTORY, name)