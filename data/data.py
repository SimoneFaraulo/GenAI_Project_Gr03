"""
Modulo per la gestione del caricamento e della pre-elaborazione del dataset CelebA.

Questo script definisce le pipeline di trasformazione delle immagini (ritaglio,
ridimensionamento, normalizzazione), inizializza il DataLoader di PyTorch
utilizzando le configurazioni globali e gestisce eventuali errori durante
il reperimento dei dati. Fornisce inoltre utilità per la manipolazione dei tensori.
"""


from config.config import BATCH_SIZE, DATA_DIRECTORY, NUM_WORKERS
from .image_dataset import CelebADataset
import torch
from torchvision.transforms import v2 as tforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 64

transform = tforms.Compose([
    tforms.ToImage(),
    tforms.CenterCrop(178), 
    tforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    tforms.ToDtype(torch.float32, scale=True)
])

try:
    data_set = CelebADataset(DATA_DIRECTORY, transform=transform)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print('Dataset CelebA caricato da:', DATA_DIRECTORY)
    print('Campioni trovati:', len(data_set))
except Exception as e:
    print(f"Errore nel caricamento del dataset: {e}")
    data_set = []
    data_loader = None

def grayscale(img_tensor, output_channels=1):
    """
    Converte un tensore di immagini RGB in scala di grigi utilizzando le trasformazioni
    funzionali di torchvision.

    Args:
        img_tensor (torch.Tensor): Il tensore dell'immagine di input (solitamente C x H x W).
        output_channels (int): Il numero di canali dell'immagine in uscita (default: 1).
    """

    return tforms.functional.rgb_to_grayscale(img_tensor, output_channels)