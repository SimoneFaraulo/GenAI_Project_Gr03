import os
import zipfile


def scompatta_tutto(percorso_cartella):
    if not os.path.exists(percorso_cartella):
        print(f"Errore: La cartella '{percorso_cartella}' non esiste.")
        return

    for file in os.listdir(percorso_cartella):
        if file.endswith(".zip"):
            path_completo = os.path.join(percorso_cartella, file)

            print(f"Sto estraendo: {file}...")

            try:
                with zipfile.ZipFile(path_completo, 'r') as zip_ref:
                    zip_ref.extractall(percorso_cartella)
                    print(f" -> Completato: {file}")
            except zipfile.BadZipFile:
                print(f" -> Errore: {file} sembra essere corrotto.")
            except Exception as e:
                print(f" -> Errore generico su {file}: {e}")


if __name__ == "__main__":
    cartella_target = r"./dataset"

    scompatta_tutto(cartella_target)
    print("\nOperazione terminata.")