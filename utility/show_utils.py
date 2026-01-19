import numpy as np
from matplotlib import pyplot as plt

def show_image(img_tensor):
    '''The parameter is a tensor in the 3xHxW format or 1xHxW format
       where H is the height of the image and W is the width
    '''
    #Normalizzazione valori dei pixel
    img_tensor=img_tensor.clip(0.0, 1.0)
    if img_tensor.shape[0]==1:
        plt.imshow(img_tensor[0], cmap='gray')
    else:
        # img_show si aspetta tensori (H, W, C)
        plt.imshow(img_tensor.permute(1,2,0))
    plt.show()


def show_images(*img_rows):
    '''Each parameter is a tensor in format Nx3xHxW or Nx1xHxW,
       where N is the number of images in the row, 
       H is the height of each image and W is the width.
       The number of parameters determine the number of rows
    '''
    img_rows=[r.clip(0.0, 1.0) for r in img_rows]
    rows=len(img_rows)
    cols=img_rows[0].shape[0]
    k=1
    # ogni riga corrisponde ad un batch di immagini
    for r in range(rows):
        # ogni colonna corrisponde ad un immagine nel batch i-esimo passato
        for c in range(cols):
            # Griglia row per cols e posizione l'immagine nella cella k
            plt.subplot(rows, cols, k)
            #Etichette da disegnare lungo quegli assi
            plt.xticks([])
            plt.yticks([])
            if img_rows[r][c].shape[0]==1:
                plt.imshow(img_rows[r][c][0], cmap='gray')
            else:
                plt.imshow(img_rows[r][c].permute(1,2,0))
            k += 1
    plt.show()

def save_images(filename, *img_rows, figsize=None):
    '''Each parameter is a tensor in format Nx3xHxW or Nx1xHxW,
       where N is the number of images in the row, 
       H is the height of each image and W is the width.
       The number of parameters determine the number of rows
    '''
    img_rows=[r.clip(0.0, 1.0) for r in img_rows]
    rows=len(img_rows)
    cols=img_rows[0].shape[0]
    k=1
    if figsize:
        #Serve a creare una figura di dimensioni precise in pollici
        plt.figure(figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows, cols, k)
            plt.xticks([])
            plt.yticks([])
            if img_rows[r][c].shape[0]==1:
                plt.imshow(img_rows[r][c][0], cmap='gray')
            else:
                plt.imshow(img_rows[r][c].permute(1,2,0))
            k += 1
    #Invece di aprire una finestra scrive su un file
    #Salva img ad alti dpi = 300
    #bbox_inches='tight' forza matplotlib a eliminare i bordi bianchi attorno l'immagine
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    #Necessario per liberare la ram associata alla figura, con plt.show lo facciamo automaticamente chiudendo la finistra
    plt.close()

def parameter_count(model):
    "Returns the number of parameters in a model"
    count=0
    for p in model.parameters():
        # p = Tensore di pesi
        count+=p.numel() #Return: numero totale di elementi nel tensore
    return count