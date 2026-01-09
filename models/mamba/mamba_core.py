import torch
from torch import nn
from .pscan import pscan

class MambaBlock(nn.Module):
    def __init__(self, dim, state_size):
        super().__init__()
        self.dim = dim
        self.state_size = state_size

        A = torch.ones(dim, state_size)
        r = torch.linspace(0.5, state_size / 2, state_size)
        A *= r[None, :]
        logA = torch.log(A)
        self.logA = nn.Parameter(logA)
        self.projB = nn.Linear(dim, state_size)
        self.projC = nn.Linear(dim, state_size)
        self.projDelta = nn.Linear(dim, 1, bias=False)
        biasDelta = torch.zeros(1, 1, dim)
        self.biasDelta = nn.Parameter(biasDelta)
        self.softplus = nn.Softplus()

    def forward(self, x):
        self.clean_cached()
        A = self.computeA()
        B, C, Delta = self.computeBCDelta(x)
        Abar, Bbar = self.discretize(Delta, A, B)
        h = self.perform_scan(Abar, Bbar, x)
        y = torch.einsum('bln,bldn->bld', C, h)
        return y

    @torch.no_grad()
    def inference_start(self, batch_size=1):
        self.cached_A = self.computeA()
        self.cached_h = torch.zeros(
            batch_size, 1, self.dim, self.state_size,
            device=self.cached_A.device)

    def clean_cached(self):
        self.cached_A = None
        self.cached_h = None

    @torch.no_grad()
    def inference_step(self, x):
        A = self.cached_A
        B, C, Delta = self.computeBCDelta(x)
        Abar, Bbar = self.discretize(Delta, A, B)
        h = Abar * self.cached_h + Bbar * x[..., None]
        y = torch.einsum('bln,bldn->bld', C, h)
        self.cached_h.copy_(h)
        return y

    def computeA(self):
        return -torch.exp(self.logA)

    def computeBCDelta(self, x):
        B = self.projB(x)
        C = self.projC(x)
        Delta = self.softplus(self.biasDelta + self.projDelta(x))
        return B, C, Delta

    def discretize(self, Delta, A, B):
        DeltaA = Delta[:, :, :, None] * A[None, None, :, :]
        Abar = torch.exp(DeltaA)
        DeltaB = Delta[:, :, :, None] * B[:, :, None, :]
        denom = DeltaA + 1e-7
        Bbar = (Abar - 1.0) / (denom.abs().clamp(min=1e-10) * denom.sign()) * DeltaB

        return Abar, Bbar

    def perform_scan(self, Abar, Bbar, x):
        Atilde = Abar
        Xtilde = Bbar * x[..., None]
        return pscan(Atilde, Xtilde)


class MambaLayer(nn.Module):
    def __init__(self, dim, state_size, conv_kernel=4,
                 expansion=1):
        super().__init__()
        edim = int(dim * expansion)
        self.dim = dim
        self.edim = edim
        self.state_size = state_size
        self.conv_kernel = conv_kernel

        self.activation = nn.SiLU()
        self.proj_1 = nn.Linear(dim, edim)
        self.conv = nn.Conv1d(edim, edim, conv_kernel, padding=conv_kernel - 1)
        self.mamba = MambaBlock(edim, state_size)
        self.proj_2 = nn.Linear(dim, edim)
        self.proj_3 = nn.Linear(edim, dim)

    def forward(self, x):
        ex = self.proj_1(x)
        ex = self.do_conv(ex)
        ex = self.activation(ex)
        y = self.mamba(ex)
        g = self.proj_2(x)
        g = self.activation(g)
        y *= g
        y = self.proj_3(y)
        return y

    @torch.no_grad()
    def inference_start(self, batch_size=1):
        self.mamba.inference_start(batch_size)
        self.cached_x = torch.zeros(batch_size, self.conv_kernel, self.edim)

    @torch.no_grad()
    def inference_step(self, x):
        if self.cached_x.device != x.device:
            self.cached_x = self.cached_x.to(device=x.device)

        ex = self.proj_1(x)
        ex = self.do_conv(ex, True)
        ex = self.activation(ex)
        y = self.mamba.inference_step(ex)
        g = self.proj_2(x)
        g = self.activation(g)
        y *= g
        y = self.proj_3(y)
        return y

    def do_conv(self, x, inference=False):
        if inference:
            ck = self.conv_kernel
            ed = self.edim
            self.cached_x = torch.cat([self.cached_x, x], dim=1)[:, 1:]
            x = self.cached_x
        L = x.shape[1]
        x = x.permute(0, 2, 1)
        y = self.conv(x)[:, :, :L]
        y = y.permute(0, 2, 1)
        if inference:
            y = y[:, -1:, :]
        return y

class ResidualMambaLayer(nn.Module):
    def __init__(self, dim, state_size, conv_kernel=4,
                 expansion=1):
        super().__init__()
        self.mamba = MambaLayer(dim, state_size, conv_kernel, expansion)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

    def inference_start(self, batch_size=1): # MODIFICA: Aggiunto parametro
        self.mamba.inference_start(batch_size) # Passo batch_size

    def inference_step(self, x):
        return x + self.mamba.inference_step(self.norm(x))