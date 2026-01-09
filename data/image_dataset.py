from PIL import Image
import os
import torch
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, folder, transform=None):
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
        if not os.path.isfile(self.attr_path):
            raise RuntimeError(f"File degli attributi non trovato in {self.attr_path}")

        with open(self.attr_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines[2:]):
            parts = line.split()
            filename = parts[0]
            all_attrs = [int(x) for x in parts[1:]]
            current_attrs = [all_attrs[idx] for idx in self.target_indices]
            
            self.filenames.append(filename)
            self.labels.append(current_attrs)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        attr_list = self.labels[index]
        target = torch.tensor(attr_list, dtype=torch.float32)

        return img, target