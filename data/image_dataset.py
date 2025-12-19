'''This module implements a simple Dataset (see torch.utils.data.Dataset)
   that loads images from an unstructured folder.
   Images will not have associated target values.
'''

from PIL import Image
import os
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    '''A Dataset that loads images from an unstructured folder.
       Images will not have associated target values.
    '''

    def __init__(self, folder, extensions=['.jpg', '.jpeg'],
                 transform=None, recurse=False, caching=False):
        '''PARAMS
           folder:  the name of the directory containing the images
           extensions:  a list of filename extensions to be considered;
                 default: ['.jpg', '.jpeg']
           transform: a transform that is applied to each image before
                 being returned by __getitem__; default: None
                 Note: you need to provide a transform if you want the
                 images converted to PyTorch tensors (see 
                 torchvision.transforms.v2.ToImage);
                 Note: the transform is re-applied each time 
                 __getitem__ is called
           recurse: if True, will scan also the subfolders of folder;
                 default: False
           caching: if True, will keep in a cache the images loaded;
                 otherwise, images are reloaded from their file each
                 time they are returned by __getitem__; 
                 default: False
        '''
        extensions=[e.lower() for e in extensions]
        folder=os.path.abspath(os.path.expanduser(folder))
        self.filenames=self.find_files(folder, extensions, recurse)
        self.transform=transform
        self.caching=caching
        self.cache={ }

    def __len__(self):
        '''Returns the number of images in the dataset'''
        return len(self.filenames)

    def __getitem__(self, index):
        '''Returns the image at index 'index' (with the transform
           applied, if present)
        '''
        img=self.get_image(index)
        if self.transform:
            img=self.transform(img)
        return img

    def find_files(self, folder, extensions, recurse):
        "Private"
        lst=[]
        for name in os.listdir(folder):
            pname=os.path.join(folder, name)
            if os.path.isfile(pname):
                n, ext=os.path.splitext(name)
                if ext.lower() in extensions:
                    lst.append(pname)
            elif recurse and os.path.isdir(pname):
                sublst=self.find_files(pname, extensions, recurse)
                lst += sublst
        return lst

    def get_image(self, index):
        "Private"
        if self.caching and index in self.cache:
            return self.cache[index]
        pname=self.filenames[index]
        img=Image.open(pname)
        if self.caching:
            self.cache[index]=img
        return img
