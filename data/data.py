from config import data_base, BATCH_SIZE
from image_dataset import ImageDataset
import torch
import torchvision
from torchvision.transforms import v2 as tforms
from torch.utils.data import DataLoader

DATASET_FOLDER=data_base('UNSPLASH')

IMAGE_SIZE=256 # Height and width of images

# Data conversion + Minimal data augmentation for images
transform=tforms.Compose([
    tforms.ToImage(),
    tforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), ratio=(1,1),
                scale=(0.7,1.0), antialias=True),
    tforms.RandomHorizontalFlip(),
    tforms.ColorJitter(brightness=0.2, contrast=0.1),
    tforms.ToDtype(torch.float32, scale=True)
    ])

data_set=ImageDataset(DATASET_FOLDER, transform=transform, caching=True)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)


def grayscale(img_tensor, output_channels=1):
    '''Assume that the input is a tensor (3xHxW) representing
       a WxH color image, or (Nx3xHxW) for a batch of N color images;
       The returned value has the same format as the input, but with
       a possibly reduced number of output channels.

       img_tensor      the input image/batch or images
       output_channels number of desired output channels (1 or 3);
                       default: 1                     
    '''
    return tforms.functional.rgb_to_grayscale(img_tensor, output_channels)
    

print('Dataset loaded from:', DATASET_FOLDER)
print('Dataset samples:', len(data_set))
