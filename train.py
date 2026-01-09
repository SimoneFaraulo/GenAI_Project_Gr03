import torch
import sys
from config.config import DEVICE, checkpoint_base, EPOCHS, MODEL_TYPE, EXP_NUM, LEARNING_RATE
from utility.checkpoint_manager import CheckpointManager
from data.data import data_loader, data_set
from utility.show_utils import parameter_count
from train.trainer import Trainer
from train.model_factory import get_model_from_env


def main():
    epochs = EPOCHS

    try:
        model, exp_name = get_model_from_env()
    except ValueError as e:
        print(e)
        sys.exit(1)
        
    model = model.to(DEVICE)
    print(f"Parametri del modello: {parameter_count(model):,}\n\n")
    checkpoint_folder = checkpoint_base(f"{exp_name}_{EXP_NUM}")
    cpm = CheckpointManager(checkpoint_folder, kept_checkpoints=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=data_loader,
        dataset=data_set,
        device=DEVICE,
        checkpoint_manager=cpm,
        checkpoint_folder=checkpoint_folder
    )

    trainer.load_checkpoint()
    trainer.train_loop(epochs=epochs, batches_per_epoch=len(data_loader))

if __name__ == '__main__':
    main()