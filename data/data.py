from config.config import BATCH_SIZE, DATA_DIRECTORY, NUM_WORKERS
from .image_dataset import CelebADataset
import torch
from torchvision.transforms import v2 as tforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 64  # Risoluzione richiesta dal progetto (64x64)

# Trasformazioni per CelebA
# 1. CenterCrop(178): Ritaglia il quadrato centrale dove si trova il volto (standard per CelebA)
# 2. Resize(IMAGE_SIZE): Ridimensiona a 64x64
transform = tforms.Compose([
    tforms.ToImage(),
    tforms.CenterCrop(178), 
    tforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    tforms.ToDtype(torch.float32, scale=True)
])

# Istanzia il nuovo dataset compatibile con la struttura locale
try:
    data_set = CelebADataset(DATA_DIRECTORY, transform=transform)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    print('Dataset CelebA caricato da:', DATA_DIRECTORY)
    print('Campioni trovati:', len(data_set))
except Exception as e:
    print(f"Errore nel caricamento del dataset: {e}")
    data_set = []
    data_loader = None

def grayscale(img_tensor, output_channels=1):
    '''Funzione di utilita' per convertire in scala di grigi se necessario'''
    return tforms.functional.rgb_to_grayscale(img_tensor, output_channels)