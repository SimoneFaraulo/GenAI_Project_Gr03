import torch
from torch import nn
import math
from config.config import *


class NoiseSchedule:
    """
    Gestisce lo schedule del rumore e pre-calcola tutti i coefficienti
    (alpha, beta, radici cumulative) necessari per il processo di diffusione forward e reverse.
    """
    def __init__(self, schedule_len=NOISE_SCHEDULE_L, s=0.008, device=DEVICE):
        """
        Inizializza lo schedule, calcolando i tensori per beta, alpha e le loro varianti cumulative su dispositivo specificato.
        
        Args:
            schedule_len (int): Lunghezza della sequenza di rumore (default da config).
            s (float): Shift parameter per lo schedule coseno (default 0.008).
            device (torch.device): Dispositivo su cui allocare i tensori.
        """
        #attributo di comodità utile per logica successiva
        self.schedule_len = schedule_len
        #genera un tensore monodimensionale di sched_len + 1 valori tra 0 e sched_len a passo lineare
        t = torch.linspace(0.0, schedule_len, schedule_len + 1, device=device) / schedule_len
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        a = f / f[0] # a_0 = 1
        #calcolo dei beta secondo la formulazione matematica escludendo a[0] e a[T+1]
        self.beta = (1 - a[1:] / a[:-1]).clip(0.0, 0.99)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)
        self.one_minus_beta = 1 - self.beta
        self.one_minus_alpha = 1 - self.alpha
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.sqrt_1_alpha = torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta = torch.sqrt(self.one_minus_beta)
        #Sono tutti vettori lineari di T elementi con Cosine Noise Schedule per ogni t

class TimeEncoding:
    """
    Implementa il 'Sinusoidal Time Embedding' per codificare l'informazione temporale (t)
    in vettori utilizzabili dalla rete neurale.
    """
    def __init__(self, dim, schedule_len=NOISE_SCHEDULE_L, device=DEVICE):
        """
        Pre-calcola la matrice degli embedding posizionali (seno e coseno) per tutti
        i passi temporali definiti dallo schedule.
            
        Args:
            dim (int): Dimensione finale del vettore di embedding.
            schedule_len (int): Numero totale di step temporali.
            device (torch.device): Dispositivo di esecuzione.
        """
        self.dim = dim
        self.schedule_len = schedule_len
        dim2 = dim // 2
        encoding = torch.zeros(schedule_len, dim, device=device)
        # Ang = Misura del tempo nella mia sinusoide
        ang = torch.linspace(0.0, torch.pi / 2, schedule_len, device=device)
        # mul = Misura della frequenza della sinusoide su ogni componente di dim
        logmul = torch.linspace(0.0, math.log(40), dim2, device=device)
        mul = torch.exp(logmul)

        for i in range(dim2):
            # ang [schedule_len ] [mul[i]] = frequenza della singola componente iesima di dim (scalare)
            a = ang * mul[i]
            # Ogni riga avrà una sinusoide a t diversa con una w diversa ad ogni componente di dim
            # sin(w * t) -> t = ang, w = mul, alte frequenze nelle componenti di dim maggiori
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        # Encoding utile a distinguere efficientemente i vari passi temporali
        # Ogni angolo (t) rappresenta l'info di ogni passo t
        # la freq (mul) è utile a rappresentare in maniera completa l'info
        # La rete tramite (t) saprà a quale passo temporale si trova e se si trova in t piccoli (vicino alla distribuzione target)
        # si concentrerà sulle componenti ad alta frequenza per ricostruire i dettagli di z viceversa nelle fasi iniziali (t grandi)
        self.encoding = encoding


    def __getitem__(self, t):
        """
        Restituisce il vettore di embedding corrispondente al passo temporale t specificato.
        
        Args:
            t (torch.Tensor | int): Indice o tensore di indici temporali.

        Returns:
            torch.Tensor: Il vettore di embedding corrispondente.
        """
        return self.encoding[t]


class UNetBlock(nn.Module):
    """
    Rappresenta un blocco ricorsivo della U-Net. Ogni blocco contiene un encoder,
    un decoder e opzionalmente un blocco interno (inner block) per gestire diverse risoluzioni.
    """
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        """
        Configura i layer di convoluzione per l'encoder, il decoder e il combinatore finale,
        oltre a istanziare il blocco interno se necessario.
        
        Args:
            size (int): Risoluzione spaziale (H/W) dell'input a questo livello.
            outer_features (int): Numero di canali in input e output del blocco.
            inner_features (int): Numero di canali intermedi (per la trasformazione interna).
            cond_features (int): Dimensione del vettore di embedding delle condizioni.
            inner_block (nn.Module, optional): Il blocco UNet annidato (livello inferiore), se presente.
        """
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        self.encoder = self.build_encoder(outer_features + cond_features, inner_features)
        #Al decoder fornisco anche le info posizionali, sarà così in grado di dosare
        #il rumore da rimuovere sulla base del t attuale
        self.decoder = self.build_decoder(inner_features + cond_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)
        self.inner = inner_block

    def forward(self, x, time_encodings, cond):
        """
        Esegue il passaggio forward: codifica l'input, passa attraverso il blocco interno (se presente),
        concatena le feature (skip connection) e decodifica l'output.
        
        Args:
            x (torch.Tensor): Tensore di input [Batch, outer_features, size, size].
            time_encodings (torch.Tensor): Tensore degli embedding temporali.
            cond (torch.Tensor): Tensore degli embedding degli attributi.

        Returns:
            torch.Tensor: Output del blocco combinato con le skip connection.
        """
        # input x0 = [B, 3, size, size]
        x0 = x
        ## cc = [B, conf_feat] -> [B, cond_feat, size, size]
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size) # matcha dimensioni dell'immagine input
        # x = [B, 3 + cond_feat, size, size]
        x = torch.cat((x, cc), dim=1)
        y = self.encoder(x)
        #ritorna un y con size // 2

        if self.inner:
            # Eseguo la ricorsione passo un input al figlio del padre
            y = self.inner(y, time_encodings, cond)

        # sono nella bottle neck -> ho ridotto la dim_outer in una di inner/2
        half_size = self.size // 2
        # preparo l'info condizionale per darla in paso al decoder che prende tensori con
        # dim spaziale / 2
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size) # matcha dimensioni dimezzate
        # t = [B, dim_time_enc] -> [B, dim_time_enc, outer_size//2, outer_size//2]
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half_size, half_size)
        # y1 = [B, dim_time_enc + cond_dim + y_feat, half_size, half_size]
        y1 = torch.cat((y, cc, tt), dim=1)
        # Ho pronto il tensore da dare in input al decoder
        x1 = self.decoder(y1)
        # x1 = [B, outer_feat, size, size]
        x2 = torch.cat((x1, x0), dim=1) # skip connection con l'input originale, i canali raddoppiano
        return self.combiner(x2)        # i canali tornano a outer_features

    def build_combiner(self, from_features, to_features):
        """Metodo helper per costruire i blocchi sequenziali di layer convoluzionali e attivazioni. 
        
        Args:
            from_features (int): Canali in ingresso.
            to_features (int): Canali in uscita.
        """
        return nn.Conv2d(from_features, to_features, 1) # conv 1x1 per ridurre i canali

    def build_encoder(self, from_features, to_features):
        """Metodo helper per costruire i blocchi sequenziali di layer convoluzionali e attivazioni. 
        
        Args:
            from_features (int): Canali in ingresso.
            to_features (int): Canali in uscita.
        """
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False), # mantiene la dimensione
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False), # dimezza la dimensione raddoppia i canali
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )

    def build_decoder(self, from_features, to_features):
        """Metodo helper per costruire i blocchi sequenziali di layer convoluzionali e attivazioni. 
        
        Args:
            from_features (int): Canali in ingresso.
            to_features (int): Canali in uscita.
        """
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False), # mantiene la dimensione
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False), # raddoppia la dimensione dimezza i canali
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )


class ConditionalDiffusion(nn.Module):
    """
    Classe principale del Modello DDPM.
    Gestisce la costruzione della U-Net, il calcolo della loss e il processo generazione.
    """
    def __init__(self):
        """
        Inizializza il modello, inclusi gli embedding degli attributi, lo schedule del rumore,
        il time encoding e la struttura U-Net ricorsiva.
        """
        super().__init__()
        self.img_size = IMG_SIZE
        self.channels = IMG_CHANNELS
        self.hidden_dims = DIFFUSION_HIDDEN_DIMS
        self.noise_schedule = NoiseSchedule(device=DEVICE)
        self.time_encoding = TimeEncoding(TIME_ENCODING_SIZE, device=DEVICE)
        #Proiezione dell'info condizionale da [B, 3] -> [B, EMBD_DIM]
        self.attr_embed = nn.Sequential(
            nn.Linear(ATTR_DIM, ATTR_EMBED_DIM),
            nn.SiLU(), #Migliore per la retropropagazione dei gradienti, ricordiamo che alcuni elementi condizionali
                       #sono -1 in ReLu avremmo attivazione < 0 e gradiente nullo, SiLu favorisce apprendimento di feature migliori
            nn.Linear(ATTR_EMBED_DIM, ATTR_EMBED_DIM),
        )
        self.pre = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_dims[0], 3, padding='same'), # porta a hidden_dims[0] (default 64) canali
            nn.ReLU()
        )
        #Costruzione della rete
        self.unet = self.build_unet(self.img_size, self.hidden_dims)
        self.post = nn.Sequential(
            nn.ReLU(),
            # porta di nuovo a 3 canali per RGB
            nn.Conv2d(self.hidden_dims[0], self.channels, 3, padding='same')
            # non c'è attivazione finale non lineare per predire il rumore
        )

    def forward(self, x, t, cond):
        """
        Propaga l'input attraverso la rete U-Net condizionata dal time-step t e dagli attributi (cond),
        restituendo il rumore predetto.
        
        Args:
            x (torch.Tensor): Batch di immagini rumorose [Batch, Channels, H, W].
            t (torch.Tensor): Batch di indici temporali [Batch].
            cond (torch.Tensor): Batch di vettori attributi (non encodati) [Batch, Attr_Dim].

        Returns:
            torch.Tensor: Il rumore predetto dalla rete.
        """
        # t_enc = [B, DIM_T_ENC]
        t_enc = self.time_encoding[t]
        # [B, 3] -> [B, DIM_EMB]
        cond_emb = self.attr_embed(cond)
        # [B, 3, H, W] -> [B, H_DIM[0], H, W]
        x_in = self.pre(x)
        y = self.unet(x_in, t_enc, cond_emb)
        # [B, H_DIM[0], H, W] -> [B, 3, H, W]
        output = self.post(y)
        return output

    def build_unet(self, size, feat_list):
        """
        Costruisce ricorsivamente l'architettura U-Net basandosi sulla lista delle dimensioni delle feature fornite.
        
        Args:
            x (torch.Tensor): Batch di immagini rumorose [Batch, Channels, H, W].
            t (torch.Tensor): Batch di indici temporali [Batch].
            cond (torch.Tensor): Batch di vettori attributi (non encodati) [Batch, Attr_Dim].

        Returns:
            torch.Tensor: Il rumore predetto dalla rete.
        """
        if len(feat_list) > 2:
            # vado a creare ogni volta un unet che riduce la dim spaziale di 2
            inner_block = self.build_unet(size // 2, feat_list[1:])
        else:
            inner_block = None

        # se la lista è solo di due elementi ho solo un outer e inner feature
        # sono nella bottleneck e inizio i return creando la rete per intero
        return UNetBlock(size, feat_list[0], feat_list[1], ATTR_EMBED_DIM, inner_block)

    def compute_loss(self, x0, cond):
        """
        Calcola la loss MSE per il training. Campiona un istante t, aggiunge rumore all'immagine
        e calcola l'errore tra il rumore aggiunto e quello predetto dalla rete.
        
        Args:
            x0 (torch.Tensor): Batch di immagini originali (senza rumore).
            cond (torch.Tensor): Batch di attributi corrispondenti.

        Returns:
            torch.Tensor: Scalare rappresentante la MSE loss.
        """
        # [B, 3, H, W]
        batch_size = x0.shape[0]
        P = 0.2
        # [B, 3]
        cond = cond.clone()  # evita di modificare l'input originale perché usiamo il dropout sugli attributi
        u = torch.rand((batch_size,), device=x0.device)
        cond[u < P, :] = 0.0 # condizionamento nullo con probabilità P
        #Seleziono un t randomico diverso per ogni B
        t = torch.randint(0, self.noise_schedule.schedule_len, (batch_size,), device=x0.device)
        # eps = [B, 3, H, W]
        eps = torch.randn_like(x0)
        # alpha è uno scalare per ogni posizione t lo vedo come un tensore replicato su tutto il batch
        # [B, 1, 1, 1]
        # Inizialmente sqrt_alpha sarà dimensione [B], con view lo rivedo in [B, 1,1,1]
        # view non fa broadcasting semplicemente riorganizza i dati in un altra forma
        # non replica assolutamente alcun dato
        sqrt_alpha = self.noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        sqrt_1_alpha = self.noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)
        # Calcolo zt con il diff kernel
        # Con broadcasting le dimensioni di alpha su C, W, H vengono replicate, replichiamo il valore
        #di alpha su tutte quelle dim così come anche per sqrt_1_alpha
        zt = sqrt_alpha * x0 + sqrt_1_alpha * eps # diffusion kernel
        # Stimo g
        g = self(zt, t, cond) #INPUT-> [B, 3, H, W], [DIM_T_ENC], [B, 3]
        loss = nn.functional.mse_loss(g, eps)
        
        return loss

    @torch.no_grad()
    def sample(self, num_samples, device, cond=None, lam=LAMBDA, steps=NOISE_SCHEDULE_L, eta=1.0):
        """
        Esegue il processo di generazione usando l'algoritmo DDIM.
        
        Args:
            num_samples (int): Numero di immagini da generare.
            device (torch.device): Dispositivo di esecuzione.
            cond (torch.Tensor, optional): Attributi specifici. Se None, vengono generati random.
            lam (float): Coefficiente di "guidance" (CFG).
            steps (int): Numero di passi DDIM (default: NOISE_SCHEDULE_L (DDPM)).
            eta (float): Parametro di stocasticità (1.0 = DDPM (default), 0.0 = deterministico/DDIM).

        Returns:
            torch.Tensor: Batch di immagini generate [num_samples, Channels, H, W].
        """
        if cond is None:
            cond = torch.randint(0, 2, (num_samples, ATTR_DIM)).float().to(device)
            cond = (cond * 2) - 1

        # [N, dim]
        cond0 = torch.zeros_like(cond).to(device)

        n = num_samples
        # [B, 3, h, w]
        z = torch.randn(n, self.channels, self.img_size, self.img_size, device=device) # z ~ N(0,I)
        was_training = self.training
        self.eval()

        # sottoinsieme di passi temporali lineari da 0 a schedule_len-1 con steps elementi
        # tensore di dimensione steps da 0 a T-1
        tau_seq = torch.linspace(0, self.noise_schedule.schedule_len - 1, steps, dtype=torch.long, device=device)

        for i in reversed(range(steps)):
            tau_curr = tau_seq[i]

            # se siamo all'ultimo step (i=0) il precedente è "tempo -1" (che corrisponde a t<0).
            tau_prev = tau_seq[i-1] if i > 0 else -1

            # [B]
            t = tau_curr.view(1).expand(n) # t espanso a batch size

            #classifier free guidance
            g1 = self(z, t, cond)  # rumore predetto con condizionamento
            g0 = self(z, t, cond0) # rumore predetto senza condizionamento
            # [B, 3, H, W]
            g = lam * g1 + (1 - lam) * g0 

            # tau_curr [B] e tau_prev B or -1
            z = self.ddim_step(z, g, eta, tau_curr, tau_prev)

        if was_training:
            self.train()
            
        return z
    
    def ddim_step(self, zt, g, eta, tau_curr, tau_prev):
        """
        Esegue un singolo passo di aggiornamento DDIM.
        Calcola z_tau_i-1 partendo da z_tau_i e dal rumore predetto g.
        
        Args:
            zt (torch.Tensor): Stato corrente z_tau_i.
            g (torch.Tensor): Rumore predetto dalla rete.
            eta (float): Parametro di stocasticità.
            tau_curr (torch.Tensor): Indice temporale corrente.
            tau_prev (torch.Tensor): Indice temporale precedente
        """
        a_curr = self.noise_schedule.alpha[tau_curr]
        
        # Recupera alpha_prev (gestendo il caso tau_prev < 0 -> alpha = 1.0)
        if tau_prev >= 0:
            #recupero l'alpha associata al tempo t-delta
            a_prev = self.noise_schedule.alpha[tau_prev]
        else:
            # Mi trovo in alpha_0 non definito e di default = 1
            a_prev = torch.tensor(1.0, device=zt.device)

        # calcolo esattamente sigma il rumore da aggiungere a z_t-delta generato
        sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_curr) * (1.0 - a_curr / a_prev))
        
        c1 = torch.sqrt(a_prev / a_curr) # coefficiente per zt
        c2 = torch.sqrt(1.0 - a_prev - sigma**2) - torch.sqrt(a_prev * (1.0 - a_curr) / a_curr) # coefficiente per g
         
        eps = torch.randn_like(zt) # rumore casuale

        #sto moltiplicando scalari per tensori, pytorch esegue in automatico
        #il broadcasting di questo scalare su ogni dimensione
        z_prev = c1 * zt + c2 * g + sigma * eps # calcolo z_tau_i-1
        
        return z_prev
    
    def diffusion_train_step(self, batch, device):
        """
        Wrapper per il passo di training: sposta i batch sul device corretto, calcola la loss
        e restituisce le metriche per il logging.
        
        Args:
            batch (tuple): Una tupla (immagini, attributi) dal dataloader.
            device (torch.device): Dispositivo di esecuzione.

        Returns:
            tuple: (loss, metrics_dict) dove metrics_dict è un dizionario per il logging.
        """
        images, attributes = batch
        # [B, 3, H, W]
        images = images.to(device)
        # [B, 3]
        attributes = attributes.to(device)
        loss = self.compute_loss(images, attributes)

        metrics = {
            'mse': loss.item()
        }

        return loss, metrics
    
    def train_step_fn(self, batch, device):
        """
        Metodo di compatibilità per essere chiamato dal Trainer generico.
        
        Args:
            batch (tuple): Batch di dati (immagini, label).
            device (torch.device): Dispositivo corrente.
        """
        return self.diffusion_train_step(batch, device)