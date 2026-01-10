'''
This module is used to simplify the saving and reloading
of "checkpoints", used to restart a long training.

You should create a CheckpointManager object at the beginning
of the program, and use the load_last_checkpoint method
to check if there is a saved checkpoint and, if so, reinitialize
the model (and possibly other objects, such as the optimizer)
from that.
Also, you must use the save_checkpoint method to periodically
save the current state of the training.


Example:
    cpm=CheckpointManager(folder='checkpoints')

    def start():
        model=... # Build the model
        optimizer=... # Build the optimizer
        epoch=0 # Initial number of training epochs
        cp=cpm.load_last_checkpoint()
        if cp is not None:
            # If there was a saved checkpoint, load
            # the state of the model and of the optimizer
            # from the checkpoint data
            model.load_state_dict(cp['model'])
            optimizer.load_state_dict(cp['optimizer'])
            epoch=cp['epoch']
        run_training(model, optimizer, epoch)

    def run_training(model, optimizer, epoch):
        ...
        while epoch<MAX_EPOCHS:
            ...
            epoch+=1
            # Within the training loop, periodically save
            # the state of the model and of the optimizer
            state={'model': model.state_dict(), 
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch }
            cpm.save_checkpoint(state)
            ...


Author: P. Foggia <pfoggia@unisa.it> 2025
'''

import torch
import os
import time

SEQUENCE_DIGITS=6
CHECKPOINT_EXTENSION=os.path.normcase('.ckp')
TEMP_EXTENSION='.tmp'

class CheckpointManager:
    '''
An object that manages the saving and retrieval of the checkpoints.
Each checkpoint is stored in a file whose name is:
         <folder>/<prefix><nnnnnn>.ckp
where: <folder> is a user-selected directory (by default, the current one)
       <prefix> is a user-selected name prefix (by default, '')
       <nnnnnn> is a 6-digits progressive number, incremented at each 
                saved checkpoint
The <folder> directory is created by the CheckpointManager, if it does not
exist.
The CheckpointManager also creates temporary files with the name:
       <folder>/<prefix><nnnnnn>.tmp

When a new checkpoint file is written, some of the older ones are deleted
(only a user-selected number of checkpoints is kept).

IMPORTANT: It is not safe to use concurrently (i.e. by two processes) 
a CheckpointManager writing to the same directory and with the same
name prefix.

IMPORTANT: The class is not thread-safe.
    '''
    def __init__(self, folder=os.curdir, prefix='', kept_checkpoints=10):
        '''PARAMS
           folder is the name of the directory where the checkpoint files
                  will be stored; default: the current directory
           prefix is the name prefix for the checkpoint files; default: ''
           kept_checkpoints is the number of checkpoint files to be kept
                  when a new checkpoint is saved; must be an int > 0
                  and < 500000, but the operation of the checkpoint
                  manager can become slow if this number is very
                  large; default: 10
        '''
        self.folder=self.normalize_folder(folder)
        self.prefix=os.path.normcase(prefix)
        self.kept_checkpoints=int(kept_checkpoints)
        assert self.kept_checkpoints>0 
        assert self.kept_checkpoints< 10**SEQUENCE_DIGITS/2
        self.last_save_time=0.0

    def load_last_checkpoint(self, **kwargs):
        '''If there is at least a checkpoint file, returns
           the contents of the last checkpoint file, as
           obtained from torch.load. Otherwise, returns None.

           PARAMS
           kwargs are keyword arguments passed to torch.load 
               (e.g. map_location='cpu')
        '''
        lst=self.list_checkpoint_files()
        if not lst:
            return None
        self.sort_checkpoint_list(lst)
        fname=lst[-1][1]
        return torch.load(fname, weights_only=False, **kwargs)


    def load_any_checkpoint(self, **kwargs):
        """
        Cerca automaticamente e carica il checkpoint più recente presente nella cartella designata.
        Scansiona la directory filtrando i file per l'estensione configurata, li ordina in base
        alla data di ultima modifica e restituisce il contenuto dell'ultimo salvataggio.

        Args:
            **kwargs: Argomenti arbitrari passati direttamente alla funzione torch.load
                      (es. map_location, weights_only, etc.).

        Returns:
            Any | None: Il contenuto del checkpoint caricato (es. dict, model state) oppure None
                        se nessun file compatibile viene trovato.
        """
        candidates = []
        for name in os.listdir(self.folder):
            if os.path.normcase(name).endswith(CHECKPOINT_EXTENSION):
                candidates.append(name)

        if not candidates:
            return None

        candidates.sort(key=lambda x: os.path.getmtime(os.path.join(self.folder, x)))
        last_checkpoint_name = candidates[-1]
        full_path = os.path.join(self.folder, last_checkpoint_name)

        return torch.load(full_path,weights_only=False,**kwargs)


    def save_checkpoint(self, data):
        '''Saves a new checkpoint, using torch.save to store the
           checkpoint data. If, after a successful saving, the number
           of checkpoint files is larger than self.kept_checkpoints,
           also deletes the older files until only self.kept_checkpoints
           remain.

           PARAMS
           data   the checkpoint data, stored on the file using torch.save
        '''
        lst=self.list_checkpoint_files()
        self.sort_checkpoint_list(lst)
        if not lst:
            index=1
        else:
            index=(lst[-1][0]+1)%(10**SEQUENCE_DIGITS)
        base_name=self.get_base_name(index)
        temp_name=base_name+TEMP_EXTENSION
        torch.save(data, temp_name)
        checkpoint_name=base_name+CHECKPOINT_EXTENSION
        os.rename(temp_name, checkpoint_name)
        to_be_deleted=len(lst)-self.kept_checkpoints+1
        for i in range(to_be_deleted):
            fname=lst[i][1]
            os.remove(fname)
        self.last_save_time=time.time()


    def delete_all_checkpoints(self):
        '''Delete all the checkpoint files.'''
        for seq, name in self.list_checkpoint_files():
            os.remove(name)

    def get_last_save_time(self):
        '''Returns the time (according to time.time()) of the
           last successful call to save_checkpoint.
           If save_checkpoint has not yet been called, returns 0.0
        '''
        return self.last_save_time

    def normalize_folder(self, folder):
        "private"
        folder=os.path.abspath(folder)
        if not os.path.exists(folder):
            os.makedirs(folder, mode=0o700)
        assert os.path.isdir(folder)
        return folder

    def list_checkpoint_files(self):
        "private"
        lst=[]
        for name in os.listdir(self.folder):
            valid, seq=self.parse_name(name)
            if valid:
                name=os.path.join(self.folder, name)
                lst.append((seq, name))
        return lst

    def parse_name(self, name):
        "private"
        name=os.path.normcase(name)
        base, ext=os.path.splitext(name)
        if ext!=CHECKPOINT_EXTENSION or not base.startswith(self.prefix):
            return False, 0
        seqpart=base[len(self.prefix):]
        if len(seqpart)!=SEQUENCE_DIGITS or not seqpart.isdigit():
            return False, 0
        return True, int(seqpart)

    def sort_checkpoint_list(self, cp_list):
        "private"
        n=len(cp_list)
        all_9=10**SEQUENCE_DIGITS-1
        if n<=1:
            return
        assert n<all_9
        cp_list.sort()
        seq_max=cp_list[-1][0]
        if seq_max!=all_9:
            return
        i=0
        offset=all_9+1
        while cp_list[i][0]==i:
            cp_list[i] = (cp_list[i][0]+offset, cp_list[i][1])
            i += 1
        cp_list.sort()
        i=-1
        while cp_list[i][0]>=offset:
            cp_list[i] = (cp_list[i][0]-offset, cp_list[i][1])
            i -= 1

    def get_base_name(self, index):
        assert index>=0 and index<10**SEQUENCE_DIGITS
        name=self.prefix+format(index, f"0{SEQUENCE_DIGITS}d")
        return os.path.join(self.folder, name)
        


