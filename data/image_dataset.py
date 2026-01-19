from PIL import Image
import os
import torch
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """
    Dataset personalizzato per il caricamento e la gestione del dataset CelebA.

    Questa classe eredita da torch.utils.data.Dataset e gestisce il caricamento
    delle immagini dalla cartella specificata e il parsing del file degli attributi,
    selezionando solo specifici target (indici 20, 31, 39).
    """

    def __init__(self, folder, transform=None):
        """
        Inizializza il dataset impostando i percorsi e caricando i metadati necessari.

        Args:
            folder (str): Il percorso della cartella radice contenente le sottocartelle delle immagini ('img_align_celeba') e il file degli attributi.
            transform (callable, optional): Una funzione o trasformazione torchvision da applicare all'immagine caricata (default: None).
        """

        self.folder = folder
        self.transform = transform
        self.img_dir = os.path.join(folder, 'img_align_celeba')
        print(f"Loading images from: {self.img_dir}")
        self.attr_path = os.path.join(folder, 'list_attr_celeba.txt')
        self.target_indices = [20, 31, 39]
        self.filenames = []
        self.labels = []
        self._load_metadata()

    def _load_metadata(self):
        """
        Metodo helper per leggere il file di testo degli attributi.

        Scorre il file 'list_attr_celeba.txt', ignorando l'intestazione, ed estrae i nomi dei file e gli attributi
        specifici definiti in `self.target_indices, popolando le liste interne.

        Raises:
            RuntimeError: Se il file degli attributi non viene trovato nel percorso atteso.
        """

        if not os.path.isfile(self.attr_path):
            raise RuntimeError(f"File degli attributi non trovato in {self.attr_path}")

        with open(self.attr_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines[2:]):
            parts = line.split()
            filename = parts[0]
            all_attrs = [int(x) for x in parts[1:]]
            current_attrs = [all_attrs[idx] for idx in self.target_indices] # seleziona solo gli attributi target
            
            self.filenames.append(filename)
            self.labels.append(current_attrs)

    def __len__(self):
        """
        Restituisce il numero totale di campioni disponibili nel dataset.
        """

        return len(self.filenames)

    def __getitem__(self, index):
        """
        Recupera un singolo campione dal dataset dato il suo indice.

        Carica l'immagine da disco, la converte in RGB, applica le trasformazioni configurate e crea il tensore delle etichette.

        Args:
            index (int): L'indice del campione da recuperare.

        Returns:
            tuple: Una coppia (img, target) dove 'img' è il tensore dell'immagine trasformata e 'target' è il tensore degli attributi (float32).
        """

        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        attr_list = self.labels[index]
        target = torch.tensor(attr_list, dtype=torch.float32)

        return img, target