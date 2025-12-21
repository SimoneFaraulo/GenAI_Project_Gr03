from torch import nn

class Skip(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        if len(layers)>1:
            self.inner=nn.Sequential(*layers)
        else:
            self.inner=layers[0]

    def forward(self, x):
        return x+self.inner(x)