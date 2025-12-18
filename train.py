import torch
#from config import DEVICE, checkpoint_base
from .utility.progress_indicator import ProgressIndicator
from .utility.checkpoint_manager import CheckpointManager
#from data import data_set, data_loader, BATCH_SIZE, grayscale
#from show_utils import save_images
import sys
import time
import os
import numpy as np

BATCHES_PER_EPOCH=2048//BATCH_SIZE
LEARNING_RATE=0.0005
CHECKPOINT_FOLDER=checkpoint_base('vit_color')


print(f'Device: {DEVICE}')
model=VisionTransformer().to(device=DEVICE)
print(f'Model size: {parameter_count(model)/(1024*1024) : .5}M')
print(f'batch_size={BATCH_SIZE}')

optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

cpm=CheckpointManager(CHECKPOINT_FOLDER, kept_checkpoints=3)
epoch_count=0

state=cpm.load_last_checkpoint(map_location='cpu')
if state:
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    epoch_count=state['epoch_count']
    print('Data loaded from previous checkpoint')

def save_checkpoint(force=False):
    state={}
    state['model']=model.state_dict()
    state['optimizer']=optimizer.state_dict()
    state['epoch_count']=epoch_count
    dt=time.time()-cpm.get_last_save_time()
    if force or dt>=5*60:
        cpm.save_checkpoint(state)
        print('Checkpoint saved')

def save_snapshot():
    global epoch_count
    fname=f"snap{epoch_count:04d}.png"
    fname=os.path.join(CHECKPOINT_FOLDER, fname)
    model.eval()

    indices=np.random.randint(0, len(data_set), 10)
    lst=[data_set[i] for i in indices]
    with torch.no_grad():
        x=torch.stack(lst).to(device=DEVICE)
        xt=grayscale(x)
        y=model(xt)
        save_images(fname, xt.cpu(), y.cpu(), x.cpu(),
                    figsize=(8.5,2.5))


progress=ProgressIndicator(BATCHES_PER_EPOCH)



def training_epoch():
    global epoch_count
    progress.start_new_epoch(epoch_count)
    ct=0
    model.train()
    for X in data_loader:
        X=X.to(device=DEVICE)
        Xt=grayscale(X)
        Y=model(Xt)
        loss=loss_function(Y,X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss=loss.cpu().item()
        progress.update(1, loss)
        save_checkpoint()
        ct+=1
        if ct>=BATCHES_PER_EPOCH:
            break
    epoch_count+=1
    if epoch_count%5==0:
        save_snapshot()



def main():
    if len(sys.argv)>1:
        epochs=int(sys.argv[1])
    else:
        epochs=1
    model.train()
    for i in range(epochs):
        training_epoch()
    if epochs>0:
        save_checkpoint(True)

if __name__=='__main__':
    main()
