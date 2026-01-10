from torch import nn

class Skip(nn.Module):
    """
    Implementa un blocco con connessione residua (skip connection).
    Questo modulo applica una trasformazione all'input tramite i sotto-layer forniti
    e somma il risultato all'input originale (operazione x + F(x)).
    """
    def __init__(self, *layers):
        """
        Inizializza il modulo residuo configurando la struttura interna.
        Gestisce automaticamente il caso di layer multipli raggruppandoli in una sequenza.

        Args:
            *layers (nn.Module): Uno o più moduli PyTorch che definiscono la trasformazione
                                 da applicare al ramo residuo.
        """
        super().__init__()
        if len(layers)>1:
            self.inner=nn.Sequential(*layers)
        else:
            self.inner=layers[0]

    def forward(self, x):
        """
        Esegue il passaggio forward applicando la logica della connessione residua.

        Args:
            x (torch.Tensor): Il tensore di input da processare.

        Returns:
            torch.Tensor: La somma element-wise tra l'input originale e l'output
                          della trasformazione interna.
        """
        return x+self.inner(x)