import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
from torchvision.transforms import v2 as tforms

# Assicuriamoci di poter importare dai moduli del progetto
sys.path.append(os.getcwd())

try:
    from image_dataset import CelebADataset
except ImportError:
    print("Errore: Impossibile importare 'ImageDataset'. Assicurati che il file 'data/image_dataset.py' esista e contenga la classe aggiornata.")
    sys.exit(1)

def show_grid(dataset, num_images=16):
    # Seleziona indici casuali
    indices = [i for i in range (16)]
    # indices = np.random.choice(len(dataset), num_images, replace=False)
    
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()
    
    # Nomi degli attributi nell'ordine restituito dal dataset (indici 20, 31, 39)
    attr_names = ["Male", "Smiling", "Young"]
    
    print(f"Visualizzazione di {num_images} immagini casuali...")
    
    for i, idx in enumerate(indices):
        img, labels = dataset[idx]
        
        # L'immagine è un tensore (C, H, W) float [0, 1]. 
        # Convertiamo in (H, W, C) per matplotlib.
        img_np = img.permute(1, 2, 0).numpy()
        
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')
        
        # Costruisce la stringa delle etichette
        # labels è un tensore [Male, Smiling, Young] con valori 0 o 1
        label_text = []
        for j, val in enumerate(labels):
            if val > 0.5:
                label_text.append(attr_names[j])
            # Opzionale: decommenta sotto se vuoi vedere anche i "Not ..."
            # else:
            #     label_text.append("Not " + attr_names[j])
        
        if not label_text:
            title = "No Attributes"
        else:
            title = "\n".join(label_text)
            
        ax.set_title(title, fontsize=9, color='blue')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    DATASET_ROOT = './dataset'  # La cartella che contiene 'Anno' e 'img_align_celeba'
    IMAGE_SIZE = 64
    # ----------------------

    if not os.path.exists(DATASET_ROOT):
        print(f"Attenzione: La cartella '{DATASET_ROOT}' non esiste.")
        print("Assicurati di aver scompattato i file zip nella cartella corretta.")
        sys.exit(1)

    # Definizione delle trasformazioni (le stesse di data.py)
    transform = tforms.Compose([
        tforms.ToImage(),
        tforms.CenterCrop(178),
        tforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        tforms.ToDtype(torch.float32, scale=True)
    ])

    # Istanzia il dataset
    try:
        dataset = CelebADataset(folder=DATASET_ROOT, transform=transform)
        print(f"Dataset caricato con successo! Totale immagini: {len(dataset)}")
        
        # Visualizza
        show_grid(dataset)
        
    except Exception as e:
        print(f"Errore durante l'inizializzazione del dataset: {e}")