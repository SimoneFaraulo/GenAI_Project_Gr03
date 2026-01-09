import argparse
import os
import sys
import torch
import time
from config.config import DEVICE, WEIGHTS_DIRECTORY
from utility.checkpoint_manager import CheckpointManager
from utility.show_utils import save_images, show_images
from train.model_factory import get_model_from_env


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script di Generazione Unificato (Singolo o Griglia)")

    # 1. Selezione Modello
    parser.add_argument('--model', type=str, required=True, choices=['vae', 'diff', 'mamba'],
                        help="Il tipo di modello da utilizzare.")

    # 2. Modalità Griglia Completa
    parser.add_argument('--all_combos', action='store_true',
                        help="Se attivo, genera una griglia con tutte le 8 combinazioni. Ignora gli attributi singoli.")

    parser.add_argument('--samples_per_combo', type=int, default=1,
                        help="[Solo per --all_combos] Numero di righe (campioni) da generare per ogni colonna (combinazione).")

    # 3. Modalità Manuale (Attributi Singoli)
    parser.add_argument('--male', type=int, choices=[0, 1], help="Genera maschio (1) o femmina (0)")
    parser.add_argument('--smiling', type=int, choices=[0, 1], help="Genera sorridente (1) o no (0)")
    parser.add_argument('--young', type=int, choices=[0, 1], help="Genera giovane (1) o no (0)")

    parser.add_argument('--num_samples', type=int, default=8,
                        help="[Solo per modalità Manuale] Numero totale di immagini da generare.")

    # 4. Output
    parser.add_argument('--show', action='store_true', help="Mostra i risultati a schermo.")
    parser.add_argument('--output_dir', type=str, default='./generated_samples', help="Cartella di output.")

    args = parser.parse_args()

    # --- VALIDAZIONE MUTUA ESCLUSIVITÀ ---
    if args.all_combos:
        if args.male is not None or args.smiling is not None or args.young is not None:
            parser.error("ERRORE: Non puoi specificare --male, --smiling o --young quando usi --all_combos.")
    else:
        if args.male is None or args.smiling is None or args.young is None:
            parser.error(
                "ERRORE: In modalità manuale devi specificare TUTTI gli attributi: --male, --smiling e --young.")

    return args


def prepare_grid_data(samples_per_combo, device):
    """Prepara il batch per le 8 combinazioni (Modalità --all_combos)"""
    # L'ordine qui definisce l'ordine delle COLONNE nel plot finale
    combinations = [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
    ]
    # Etichette brevi per la legenda delle colonne
    labels = [
        "F/NoS/Old", "F/NoS/Yng", "F/Smi/Old", "F/Smi/Yng",
        "M/NoS/Old", "M/NoS/Yng", "M/Smi/Old", "M/Smi/Yng"
    ]

    full_cond_list = []
    # Generiamo: Tutti i campioni della combo 1, poi tutti quelli della combo 2, ecc.
    for combo in combinations:
        for _ in range(samples_per_combo):
            full_cond_list.append(combo)

    cond = torch.tensor(full_cond_list, dtype=torch.float32, device=device)
    cond = (cond * 2) - 1  # Scala a [-1, 1]
    return cond, labels


def prepare_manual_data(num_samples, args, device):
    """Prepara il batch per attributi specifici (Modalità Manuale)"""
    cond = torch.zeros((num_samples, 3), device=device)
    cond[:, 0] = float(args.male)
    cond[:, 1] = float(args.smiling)
    cond[:, 2] = float(args.young)

    cond = (cond * 2) - 1  # Scala a [-1, 1]

    l = []
    l.append("M" if args.male else "F")
    l.append("Smile" if args.smiling else "NoSmile")
    l.append("Young" if args.young else "Old")
    label = "_".join(l)

    return cond, label


def main():
    args = parse_arguments()

    print(f"--- Avvio Generazione: {args.model.upper()} ---")
    mode_str = "Griglia Completa (Colonne)" if args.all_combos else "Manuale"
    print(f"Modalità: {mode_str}")

    # 1. Caricamento Modello
    try:
        model, exp_name_prefix = get_model_from_env(args.model)
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Errore init modello: {e}")
        sys.exit(1)

    # 2. Caricamento Checkpoint
    checkpoint_folder_name = f"{args.model}"
    full_checkpoint_path = os.path.join(WEIGHTS_DIRECTORY, checkpoint_folder_name)

    if not os.path.exists(full_checkpoint_path):
        full_checkpoint_path = os.path.join(WEIGHTS_DIRECTORY, f"{exp_name_prefix}_1")
        if not os.path.exists(full_checkpoint_path):
            print(f"ERRORE: Cartella checkpoint non trovata in {WEIGHTS_DIRECTORY}")
            sys.exit(1)

    print(f"Loading checkpoint from: {full_checkpoint_path}")
    cpm = CheckpointManager(folder=full_checkpoint_path)
    state = cpm.load_last_checkpoint(map_location=DEVICE)
    if not state:
        print("ERRORE: Nessun file .ckp valido trovato.")
        sys.exit(1)
    model.load_state_dict(state['model'])
    print(f"-> Pesi caricati (Epoca: {state.get('epoch_count', '?')})")

    # 3. Preparazione Input
    if args.all_combos:
        # Nota: labels qui corrisponderanno alle colonne
        cond, labels = prepare_grid_data(args.samples_per_combo, DEVICE)
    else:
        cond, label_str = prepare_manual_data(args.num_samples, args, DEVICE)

    # 4. Generazione
    print(f"Generating {cond.shape[0]} samples...")
    start = time.time()
    with torch.no_grad():
        try:
            generated_images = model.sample(cond.shape[0], DEVICE, cond=cond)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("ERRORE: GPU OOM. Riduci il numero di sample.")
                sys.exit(1)
            raise e
    print(f"Done in {time.time() - start:.2f}s")

    # 5. Salvataggio e Manipolazione Layout
    output_folder = os.path.join(args.output_dir, args.model)
    os.makedirs(output_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    generated_images = generated_images.cpu()

    if args.all_combos:
        # LOGICA DI TRASPOSIZIONE (Da Righe a Colonne)

        # 1. Reshape in [8 Combinazioni, N Samples, C, H, W]
        #    Attualmente il batch è ordinato: [Tutti Combo1, Tutti Combo2, ...]
        reshaped_imgs = generated_images.view(8, args.samples_per_combo, *generated_images.shape[1:])

        # 2. Permute (Scambiamo dim 0 e 1) -> [N Samples, 8 Combinazioni, C, H, W]
        transposed_imgs = reshaped_imgs.permute(1, 0, 2, 3, 4)

        # 3. Creazione Righe per save_images
        #    Ogni riga 'i' conterrà l'i-esimo campione di OGNI combinazione (colonna)
        rows = []
        for i in range(args.samples_per_combo):
            rows.append(transposed_imgs[i])  # Prende slice (8, C, H, W)

        print("\nLegenda Colonne (da sinistra a destra):")
        for idx, lbl in enumerate(labels):
            print(f"Col {idx + 1}: {lbl}")

        filename = f"grid_cols_{timestamp}.png"
        filepath = os.path.join(output_folder, filename)

        # figsize: Larghezza = 8 colonne, Altezza = N campioni
        # Moltiplichiamo per 2 per dare spazio
        save_images(filepath, *rows, figsize=(16, args.samples_per_combo * 2))

        if args.show:
            print("Visualizzazione griglia per colonne...")
            show_images(*rows)

    else:
        # Salvataggio MANUALE (Standard)
        filename = f"gen_{label_str}_{timestamp}.png"
        filepath = os.path.join(output_folder, filename)
        save_images(filepath, generated_images, figsize=(10, 4))

        if args.show:
            print("Visualizzazione sample...")
            show_images(generated_images)

    print(f"Risultato salvato in: {filepath}")


if __name__ == "__main__":
    main()