import time
import torch
from utility.progress_indicator import ProgressIndicator
from utility.training_utility import save_training_checkpoint, save_snapshot
from config.config import CKP_INTERVAL, FORCE_CKP, PROGRESS_INTERVAL

class Trainer:
    def __init__(self, model, optimizer, train_loader, dataset, device, 
                 checkpoint_manager, train_step_fn, checkpoint_folder):
        """
        Args:
            train_step_fn: Funzione che accetta (model, batch, device) e 
                           ritorna (loss, log_dict).
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.dataset = dataset # Serve per gli snapshot
        self.device = device
        self.cpm = checkpoint_manager
        self.train_step_fn = train_step_fn
        self.checkpoint_folder = checkpoint_folder
        self.epoch_count = 0

    def load_checkpoint(self):
        state = self.cpm.load_last_checkpoint(map_location='cpu')
        if state:
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.epoch_count = state['epoch_count']
            print(f'Resumed training from epoch {self.epoch_count}')
        else:
            print('Starting training from scratch')

    def train_loop(self, epochs, batches_per_epoch):
        progress = ProgressIndicator(batches_per_epoch, message_period=PROGRESS_INTERVAL)
        
        target_epoch = self.epoch_count + epochs
        print(f"Training started. Target epoch: {target_epoch}")

        while self.epoch_count < target_epoch:
            progress.start_new_epoch(self.epoch_count)
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                # 1. Step di training specifico del modello
                loss, log_metrics = self.train_step_fn(self.model, batch, self.device)
                
                # 2. Backpropagation generica
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                
                # 3. Logging e Checkpoint periodico
                progress.update(1, batch_loss=loss.item(), **log_metrics)
                save_training_checkpoint(self.cpm, self.model, self.optimizer, self.epoch_count, force=FORCE_CKP, interval=CKP_INTERVAL)
                
                if i >= batches_per_epoch:
                    break
            
            self.epoch_count += 1
            
            # 4. Snapshot di fine epoca
            save_snapshot(self.model, self.dataset, self.checkpoint_folder, 
                          self.epoch_count, self.device)

        # Salvataggio finale forzato
        save_training_checkpoint(self.cpm, self.model, self.optimizer, 
                                 self.epoch_count, force=True)
        print("Training finished.")