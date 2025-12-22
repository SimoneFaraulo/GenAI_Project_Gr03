from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F  # Import necessario per one_hot


class CelebADataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        # Percorsi
        self.img_dir = os.path.join(folder, 'img_align_celeba')
        print(f"Loading images from: {self.img_dir}")
        self.attr_path = os.path.join(folder, 'list_attr_celeba.txt')

        # Indici: 20: Male, 31: Smiling, 39: Young
        self.target_indices = [20, 31, 39]

        self.filenames = []
        self.labels = []  # Conterrà interi da 0 a 7

        self._load_metadata()

    def _load_metadata(self):
        if not os.path.isfile(self.attr_path):
            raise RuntimeError(f"File degli attributi non trovato in {self.attr_path}")

        with open(self.attr_path, 'r') as f:
            lines = f.readlines()

        # Salta le prime 2 righe di header
        for i, line in enumerate(lines[2:]):
            parts = line.split()
            filename = parts[0]

            # Parsing attributi
            # Leggiamo direttamente i valori raw (-1 o 1) dal file txt
            all_attrs = [int(x) for x in parts[1:]]

            # # Seleziona attributi target [-1, 1] e converti a [0, 1]
            # # selected_attrs sarà una lista tipo [0, 1, 1] (Femmina, Smiling, Young)
            # selected_attrs = [(all_attrs[idx] + 1) // 2 for idx in self.target_indices]

            # # --- NUOVA CODIFICA ---
            # # Convertiamo la lista binaria [bit2, bit1, bit0] in un intero decimale.
            # # Esempio: [Male, Smiling, Young] -> [1, 0, 1] -> 1*4 + 0*2 + 1*1 = 5
            # # Formula: v[0]*4 + v[1]*2 + v[2]*1
            # label_idx = selected_attrs[0] * 4 + selected_attrs[1] * 2 + selected_attrs[2] * 1

            # self.filenames.append(filename)
            # self.labels.append(label_idx)
            
            # --- MODIFICA CODIFICA NON ONEHOT ---
            # Estraiamo solo i 3 attributi di interesse.
            # CelebA usa nativamente -1 e 1, perfetto per noi.
            current_attrs = [all_attrs[idx] for idx in self.target_indices]
            
            self.filenames.append(filename)
            self.labels.append(current_attrs) # Salviamo il vettore [x, y, z]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)

        # Caricamento immagine
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # # Recupera l'indice della classe (es. 5)
        # label_idx = self.labels[index]

        # # Creazione One-Hot Vector
        # # num_classes=8 perché abbiamo 2^3 combinazioni
        # target_one_hot = F.one_hot(torch.tensor(label_idx), num_classes=8)

        # # Convertiamo in float32 perché la rete si aspetta float, non long
        # target = target_one_hot.float()
        
        # --- MODIFICA: Restituzione Vettore Float ---
        # Recuperiamo la lista [-1, 1, ...] salvata
        attr_list = self.labels[index]
        
        # Convertiamo in tensore Float
        # IMPORTANTE: float32 serve per la rete neurale
        target = torch.tensor(attr_list, dtype=torch.float32) 
        
        # Nota: Non serve più F.one_hot. 
        # Restituiamo direttamente il vettore [-1.0, 1.0, -1.0]

        return img, target