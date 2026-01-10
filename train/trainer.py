import torch
from utility.progress_indicator import ProgressIndicator
from utility.training_utility import save_training_checkpoint, save_snapshot
from config.config import CKP_INTERVAL, FORCE_CKP, PROGRESS_INTERVAL

class Trainer:
    """
    Gestisce l'intero ciclo di vita dell'addestramento: esegue il training loop,
    coordina il salvataggio dei checkpoint, gestisce il logging tramite ProgressIndicator
    e genera snapshot visivi periodici.
    """
    def __init__(self, model, optimizer, train_loader, dataset, device, 
                 checkpoint_manager, checkpoint_folder):
        """
        Inizializza il Trainer collegando il modello, i dati e i gestori di stato.

        Args:
            model (nn.Module): Il modello neurale da addestrare.
            optimizer (torch.optim.Optimizer): L'ottimizzatore configurato per l'aggiornamento dei pesi.
            train_loader (DataLoader): DataLoader che fornisce i batch di training.
            dataset (Dataset): Il dataset completo (utilizzato per estrarre campioni di riferimento per gli snapshot).
            device (str | torch.device): Il dispositivo di calcolo ('cpu' o 'cuda').
            checkpoint_manager (CheckpointManager): Oggetto per la gestione della persistenza (salvataggio/caricamento).
            checkpoint_folder (str): Percorso della directory in cui salvare le immagini di snapshot generate.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.dataset = dataset # Serve per gli snapshot
        self.device = device
        self.cpm = checkpoint_manager
        self.checkpoint_folder = checkpoint_folder
        self.epoch_count = 0

    def load_checkpoint(self):
        """
        Tenta di ripristinare lo stato dell'addestramento precedente.
        Carica i pesi del modello, lo stato dell'ottimizzatore e il numero di epoca corrente
        utilizzando il CheckpointManager. Se non trova nulla, inizia da zero.
        """
        state = self.cpm.load_last_checkpoint(map_location='cpu')
        if state:
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.epoch_count = state['epoch_count']
            print(f'Resumed training from epoch {self.epoch_count}')
        else:
            print('Starting training from scratch')

    def train_loop(self, epochs, batches_per_epoch):
        """
        Avvia il ciclo principale di addestramento. Itera per il numero di epoche richiesto,
        eseguendo forward/backward pass e salvando periodicamente lo stato.

        Args:
            epochs (int): Numero di epoche da eseguire a partire dallo stato attuale.
            batches_per_epoch (int): Numero massimo di batch da processare per ogni epoca (utile per troncare epoche molto lunghe).
        """
        progress = ProgressIndicator(batches_per_epoch, message_period=PROGRESS_INTERVAL)
        
        target_epoch = self.epoch_count + epochs
        print(f"Training started. Target epoch: {target_epoch}")

        while self.epoch_count < target_epoch:
            progress.start_new_epoch(self.epoch_count)
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                loss, log_metrics = self.model.train_step_fn(batch, self.device)
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                progress.update(1, batch_loss=loss.item(), **log_metrics)
                save_training_checkpoint(self.cpm, self.model, self.optimizer, self.epoch_count, force=FORCE_CKP, interval=CKP_INTERVAL)
                
                if i >= batches_per_epoch:
                    break
            
            self.epoch_count += 1

            save_snapshot(self.model, self.dataset, self.checkpoint_folder, 
                          self.epoch_count, self.device)

        save_training_checkpoint(self.cpm, self.model, self.optimizer, 
                                 self.epoch_count, force=True)
        print("Training finished.")