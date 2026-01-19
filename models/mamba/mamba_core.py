import torch
from torch import nn
from .pscan import pscan

class MambaBlock(nn.Module):
    """
    Implementa il blocco Mamba centrale.
    Questa classe gestisce la logica del Selective State Space Model (SSM), inclusa la proiezione
    dei parametri input-dipendenti (B, C, Delta), la discretizzazione dei parametri continui
    e l'esecuzione della scan parallela o passo-passo.
    """
    def __init__(self, dim, state_size):
        """
        Inizializza i parametri del blocco Mamba, incluse le proiezioni lineari e i parametri SSM.

        Args:
            dim (int): Dimensione della feature di input (D).
            state_size (int): Dimensione dello stato latente SSM (N).
        """
        super().__init__()
        self.dim = dim # D dimensione embedding
        self.state_size = state_size # N dimensione stato 

        A = torch.ones(dim, state_size)
        r = torch.linspace(0.5, state_size / 2, state_size) # va da 0.5 a N/2 con N passi
        # inizializza A in modo che ogni dimensione dell'embedding abbia gli stessi valori crescenti
        # i primi valori più piccoli comportano memoria a lungo termine mentre i valori più grandi a breve termine
        A *= r[None, :]   # simile a HiPPO
        logA = torch.log(A) 
        self.logA = nn.Parameter(logA)
        
        self.projB = nn.Linear(dim, state_size) # sB: Lienar(D,N)
        self.projC = nn.Linear(dim, state_size) # sC: Linear(D,N)
        self.projDelta = nn.Linear(dim, 1, bias=False) # sDelta: Linear(D,1) non (D,D) per semplicità
        biasDelta = torch.zeros(1, 1, dim)  # bias diverso per ogni D, ma condiviso da tutti gli stati N
        self.biasDelta = nn.Parameter(biasDelta)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Esegue il passaggio forward standard (training mode) processando l'intera sequenza in parallelo.

        Args:
            x (torch.Tensor): Tensore di input di dimensione (Batch, Seq_Len, Dim).

        Returns:
            torch.Tensor: Tensore di output processato (Batch, Seq_Len, Dim).
        """
        self.clean_cached()
        A = self.computeA() # calcola A da logA matrice garantendo autovalori negativi
        B, C, Delta = self.computeBCDelta(x)  
        Abar, Bbar = self.discretize(Delta, A, B) # ZOH
        h = self.perform_scan(Abar, Bbar, x) # sequenza di stati nascosti (B, L, D, N)
        y = torch.einsum('bln,bldn->bld', C, h)  # moltiplica C: (B,L,N) con h: (B,L,D,N) sommando su N -> (B,L,D)
        return y

    @torch.no_grad()
    def inference_start(self, batch_size=1):
        """
        Inizializza la cache necessaria per la generazione autoregressiva (inference mode).
        Precalcola il parametro A costante e azzera lo stato nascosto h.

        Args:
            batch_size (int): Dimensione del batch per l'inferenza. Default: 1.
        """
        self.cached_A = self.computeA()
        self.cached_h = torch.zeros(batch_size, 1, self.dim, self.state_size,
                                    device=self.cached_A.device)

    def clean_cached(self):
        """
        Pulisce la cache utilizzata durante l'inferenza.
        """
        self.cached_A = None
        self.cached_h = None

    @torch.no_grad()
    def inference_step(self, x):
        """
        Esegue un singolo passo di inferenza autoregressiva aggiornando lo stato nascosto corrente.

        Args:
            x (torch.Tensor): Input del passo corrente (Batch, 1, Dim).

        Returns:
            torch.Tensor: Output del passo corrente (Batch, 1, Dim).
        """
        A = self.cached_A
        B, C, Delta = self.computeBCDelta(x)
        Abar, Bbar = self.discretize(Delta, A, B)
        h = Abar * self.cached_h + Bbar * x[..., None]
        y = torch.einsum('bln,bldn->bld', C, h) # moltiplica C: (B,1,N) con h: (B,1,D,N) sommando su N -> (B,1,D)
        self.cached_h.copy_(h)
        return y

    def computeA(self):
        """
        Calcola la matrice discretizzata A basandosi sul parametro logaritmico apprendibile.
        Mantiene la stabilità numerica lavorando nello spazio logaritmico.

        Returns:
            torch.Tensor: Il parametro A.
        """
        return -torch.exp(self.logA) # garantisce autovalori negativi

    def computeBCDelta(self, x):
        """
        Calcola i parametri dinamici B, C e Delta in funzione dell'input x.

        Args:
            x (torch.Tensor): Input corrente.

        Returns:
            tuple: Una tupla contenente i tensori (B, C, Delta).
        """
        B = self.projB(x) # (B, L, N)
        C = self.projC(x) # (B, L, N)
        Delta = self.softplus(self.biasDelta + self.projDelta(x)) # (B, L, D)
        return B, C, Delta

    def discretize(self, Delta, A, B):
        """
        Converte i parametri continui del sistema dinamico in parametri discreti
        utilizzando l'approssimazione Zero-Order Hold (ZOH).

        Args:
            Delta (torch.Tensor): Parametro di passo temporale dinamico.
            A (torch.Tensor): Parametro di stato continuo.
            B (torch.Tensor): Parametro di input continuo.

        Returns:
            tuple: I parametri discretizzati (Abar, Bbar).
        """
        DeltaA = Delta[:, :, :, None] * A[None, None, :, :] # Delta: (B, L, D, 1), A: (1, 1, D, N)
        Abar = torch.exp(DeltaA)
        DeltaB = Delta[:, :, :, None] * B[:, :, None, :] # B: (B, L, 1, N)
        denom = DeltaA + 1e-7
        Bbar = (Abar - 1.0) / (denom.abs().clamp(min=1e-10) * denom.sign()) * DeltaB # (Delta * A)^-1 * (Abar - I) * B

        return Abar, Bbar

    def perform_scan(self, Abar, Bbar, x):
        """
        Esegue l'operazione di selective scan parallela (prefix scan) per un calcolo efficiente su GPU.

        Args:
            Abar (torch.Tensor): Parametro di stato discretizzato.
            Bbar (torch.Tensor): Parametro di input discretizzato.
            x (torch.Tensor): Input della sequenza.

        Returns:
            torch.Tensor: Sequenza degli stati nascosti calcolati.
        """
        Atilde = Abar
        Xtilde = Bbar * x[..., None] # (B, L, D, N)
        return pscan(Atilde, Xtilde)


class MambaLayer(nn.Module):
    """
    Implementa un layer completo Mamba che combina convoluzione locale,
    il blocco SSM (MambaBlock) e un meccanismo di Gated MLP.
    """
    def __init__(self, dim, state_size, conv_kernel=4, expansion=1):
        """
        Inizializza il layer Mamba configurando l'espansione delle dimensioni e i layer convoluzionali.

        Args:
            dim (int): Dimensione del modello in ingresso.
            state_size (int): Dimensione dello stato interno SSM.
            conv_kernel (int): Dimensione del kernel per la convoluzione 1D locale. Default: 4.
            expansion (int): Fattore di espansione per la dimensione interna del blocco. Default: 1.
        """
        super().__init__()
        edim = int(dim * expansion) # dimensione espansa interna
        self.dim = dim
        self.edim = edim
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.activation = nn.SiLU()
        
        # ramo principale: proiezione -> conv -> SSM
        self.proj_1 = nn.Linear(dim, edim)
        self.conv = nn.Conv1d(edim, edim, conv_kernel, padding=conv_kernel - 1)
        self.mamba = MambaBlock(edim, state_size)
        
        # ramo gating
        self.proj_2 = nn.Linear(dim, edim)
        
        # proiezione finale
        self.proj_3 = nn.Linear(edim, dim) 

    def forward(self, x):
        """
        Passaggio forward che applica proiezione, convoluzione, attivazione, blocco SSM e gating.

        Args:
            x (torch.Tensor): Input del layer (Batch, Seq_Len, Dim).

        Returns:
            torch.Tensor: Output del layer (Batch, Seq_Len, Dim).
        """
        # ramo principale
        ex = self.proj_1(x)
        ex = self.do_conv(ex)
        ex = self.activation(ex)
        y = self.mamba(ex)
        
        # ramo gating
        g = self.proj_2(x)
        g = self.activation(g)
        
        y *= g
        y = self.proj_3(y)
        return y

    @torch.no_grad()
    def inference_start(self, batch_size=1):
        """
        Prepara il layer per l'inferenza inizializzando sia il MambaBlock interno
        sia il buffer per la convoluzione causale.

        Args:
            batch_size (int): Dimensione del batch. Default: 1.
        """
        self.mamba.inference_start(batch_size)
        self.cached_x = torch.zeros(batch_size, self.conv_kernel, self.edim)

    @torch.no_grad()
    def inference_step(self, x):
        """
        Esegue un passo di inferenza gestendo manualmente il buffer della convoluzione
        e chiamando il passo del blocco Mamba interno.

        Args:
            x (torch.Tensor): Input corrente (Batch, 1, Dim).

        Returns:
            torch.Tensor: Output calcolato.
        """
        if self.cached_x.device != x.device:
            self.cached_x = self.cached_x.to(device=x.device)

        # ramo principale
        ex = self.proj_1(x)
        ex = self.do_conv(ex, inference=True) # usa il buffer per la convoluzione causale
        ex = self.activation(ex)
        y = self.mamba.inference_step(ex) # passo del Mamba Block
        
        # ramo gating
        g = self.proj_2(x)
        g = self.activation(g)
        
        y *= g
        y = self.proj_3(y)
        return y

    def do_conv(self, x, inference=False):
        """
        Esegue la convoluzione causale 1D. Durante l'inferenza gestisce un buffer scorrevole
        per simulare la convoluzione passo dopo passo senza accesso al futuro.

        Args:
            x (torch.Tensor): Input da convolvere.
            inference (bool): Flag per indicare se siamo in modalità generazione. Default: False.

        Returns:
            torch.Tensor: Risultato della convoluzione.
        """
        if inference:
            ck = self.conv_kernel
            ed = self.edim
            self.cached_x = torch.cat([self.cached_x, x], dim=1)[:, 1:] # mantiene gli ultimi ck elementi
            x = self.cached_x
        L = x.shape[1]
        x = x.permute(0, 2, 1) # (B, L, D) -> (B, D, L), D sono i canali
        y = self.conv(x)[:, :, :L] # prendo solo i primi L elementi per mantenere la dimensione
        y = y.permute(0, 2, 1) # (B, D, L) -> (B, L, D)
        if inference:
            y = y[:, -1:, :] # solo l'ultimo elemento per l'inferenza 
        return y

class ResidualMambaLayer(nn.Module):
    """
    Wrapper che applica una connessione residua e la normalizzazione (RMSNorm/LayerNorm)
    attorno al MambaLayer.
    """
    def __init__(self, dim, state_size, conv_kernel=4, expansion=1):
        """
        Inizializza il blocco residuo componendo il MambaLayer e la LayerNorm.

        Args:
            dim (int): Dimensione del modello.
            state_size (int): Dimensione dello stato SSM.
            conv_kernel (int): Kernel della convoluzione interna. Default: 4.
            expansion (int): Fattore di espansione. Default: 1.
        """
        super().__init__()
        self.mamba = MambaLayer(dim, state_size, conv_kernel, expansion)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Applica la normalizzazione, il layer Mamba e somma l'input originale (skip connection).

        Args:
            x (torch.Tensor): Input del blocco.

        Returns:
            torch.Tensor: Output con residuo sommato.
        """
        return x + self.mamba(self.norm(x))

    def inference_start(self, batch_size=1):
        """
        Propaga il segnale di inizio inferenza al layer Mamba sottostante.

        Args:
            batch_size (int): Dimensione del batch. Default: 1.
        """
        self.mamba.inference_start(batch_size)

    def inference_step(self, x):
        """
        Esegue un passo di inferenza applicando normalizzazione, MambaLayer e connessione residua.

        Args:
            x (torch.Tensor): Input del passo corrente.

        Returns:
            torch.Tensor: Output del passo corrente.
        """
        return x + self.mamba.inference_step(self.norm(x))