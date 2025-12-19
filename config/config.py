import os

DEVICE=os.getenv('TORCH_DEVICE', 'cuda')
HOME_DIRECTORY=os.getenv('HOME', '.')
CHECKPOINT_DIRECTORY=os.getenv('CHECKPOINT_DIRECTORY', None)
if not CHECKPOINT_DIRECTORY:
    CHECKPOINT_DIRECTORY=os.path.join(HOME_DIRECTORY, 'temp', 'CHECKPOINTS')
DATA_DIRECTORY=os.getenv('DATA_DIRECTORY', '.')
BATCH_SIZE=int(os.getenv('BATCH_SIZE', '32'))

def checkpoint_base(name):
    return os.path.join(CHECKPOINT_DIRECTORY, name)

def data_base(name):
    return os.path.join(DATA_DIRECTORY, name)
