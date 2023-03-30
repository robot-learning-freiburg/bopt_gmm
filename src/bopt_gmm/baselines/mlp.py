import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {str(nn.ReLU) : nn.ReLU,
               str(nn.SiLU) : nn.SiLU}


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=2, hidden_dim=256, activation=nn.ReLU, parameters=None, device='cuda:0'):
        super().__init__()

        if type(hidden_dim) not in {list, tuple}:
            dims = [(in_dim, hidden_dim)] + [(hidden_dim, hidden_dim)] * hidden_layers + [(hidden_dim, out_dim)]
        else:
            dims = list(zip([in_dim] + hidden_dim + hidden_dim[:1], hidden_dim + hidden_dim[-1:] + [out_dim]))
        print(dims)
        self._model = nn.Sequential(*([nn.Linear(dims[0][0], dims[0][1]).to(device)] 
                                      + sum([[activation().to(device), nn.Linear(di, do).to(device)] for di, do in dims[1:]], [])))

        if parameters is not None:
            self._model.load_state_dict(parameters)

        self.input_dim      = in_dim
        self.output_dim     = out_dim
        self._hidden_layers = hidden_layers
        self._hidden_dim    = hidden_dim
        self._activation    = activation

    def forward(self, x):
        return self._model(x)

    @classmethod
    def deserialize(cls, d, device='cuda:0'):
        d['activation'] = ACTIVATIONS[d['activation']]
        return cls(**d, device=device)
    
    def serialize(self):
        return {'in_dim'        : self.input_dim,
                'out_dim'       : self.output_dim,
                'hidden_layers' : self._hidden_layers,
                'hidden_dim'    : self._hidden_dim,
                'activation'    : str(self._activation),
                'parameters'    : self._model.state_dict()}
    
    @classmethod
    def load_model(cls, path, device='cuda:0'):
        return cls.deserialize(torch.load(path), device=device)


    def save_model(self, path):
        torch.save(self.serialize(), path)
