import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self, input_dim=1, layers=3, neurons=20,output_dim=1):
        super(PINN, self).__init__()
        
        assert input_dim > 0, "La dimensió d'entrada ha de ser positiva"
        assert layers >= 2, "El nombre de capes ha de ser almenys 2 (1 capa d'entrada i 1 capa de sortida)"
        assert neurons > 0, "El nombre de neurones per capa ha de ser positiu"
        layer_list = []
        layer_list.append(nn.Linear(input_dim, neurons))
        layer_list.append(nn.Tanh()) 
        for _ in range(layers - 1):
            layer_list.append(nn.Linear(neurons, neurons))
            layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(neurons, output_dim))
        self.net = nn.Sequential(*layer_list)

    def forward(self, t):
        return self.net(t)
    
