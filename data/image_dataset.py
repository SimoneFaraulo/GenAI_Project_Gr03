from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    '''Dataset per CelebA che carica immagini e attributi specifici.
       Restituisce (immagine, attributi).
    '''

    def __init__(self, folder, transform=None, split='train'):
        '''
        PARAMS
           folder: directory root contenente 'img_align_celeba' e 'Anno'
           transform: trasformazioni da applicare alle immagini
           split: 'train', 'val', 'test' o 'all' (per ora carica tutto se non specificato diversamente)
        '''
        self.folder = folder
        self.transform = transform
        
        # Percorsi basati sulla struttura fornita
        self.img_dir = os.path.join(folder, 'img_align_celeba')
        print(self.img_dir)
        self.attr_path = os.path.join(folder, 'list_attr_celeba.txt')
        
        # Indici degli attributi richiesti dal PDF:
        # 20: Male, 31: Smiling, 39: Young
        # (Nota: gli indici sono 0-based rispetto alla lista di attributi nel file)
        self.target_indices = [20, 31, 39]
        
        self.filenames = []
        self.labels = []
        
        self._load_metadata()

    def _load_metadata(self):
        if not os.path.isfile(self.attr_path):
            raise RuntimeError(f"File degli attributi non trovato in {self.attr_path}")

        with open(self.attr_path, 'r') as f:
            lines = f.readlines()
            
        # La prima riga è il numero di immagini, la seconda i nomi degli attributi
        # Dalla terza riga in poi ci sono i dati
        # Format: 000001.jpg -1  1 -1 ...
        
        for i, line in enumerate(lines[2:]):
            parts = line.split()
            filename = parts[0]
            
            # Parsing degli attributi (da stringa a intero)
            # Gli attributi sono valori -1 o 1. Li convertiamo spesso in 0 e 1 per PyTorch.
            # Qui mantengo il valore originale o lo converto a 0/1: (val + 1) // 2
            all_attrs = [int(x) for x in parts[1:]]
            
            # Seleziona solo gli attributi richiesti (Male, Smiling, Young)
            selected_attrs = [all_attrs[idx] for idx in self.target_indices]
            
            # Conversione in 0/1 (opzionale ma consigliata per CrossEntropy/BCE)
            selected_attrs = [(x + 1) // 2 for x in selected_attrs]

            self.filenames.append(filename)
            self.labels.append(selected_attrs)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        '''Restituisce l'immagine trasformata e il vettore degli attributi'''
        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)
        
        # Caricamento immagine
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Caricamento attributi come tensore float
        target = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return img, target