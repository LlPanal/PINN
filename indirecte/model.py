import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=3, layers=6, neurons=64, output_dim=2, 
                 params_to_learn=None, fixed_params=None):
        super(PINN, self).__init__()
        
        assert input_dim > 0, "La dimensió d'entrada ha de ser positiva"
        assert layers >= 2, "El nombre de capes ha de ser almenys 2"
        assert neurons > 0, "El nombre de neurones per capa ha de ser positiu"
        
        layer_list = []
        layer_list.append(nn.Linear(input_dim, neurons))
        layer_list.append(nn.Tanh()) 
        for _ in range(layers - 1):
            layer_list.append(nn.Linear(neurons, neurons))
            layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(neurons, output_dim))
        self.net = nn.Sequential(*layer_list)

        if params_to_learn is None:
            params_to_learn = {}
        if fixed_params is None:
            fixed_params = {}

        all_param_names = ['alpha', 'beta', 'mu', 'd1', 'd2']

        for name in all_param_names:
            if name in params_to_learn:
                # Si l'hem de descobrir, el creem com a nn.Parameter (entrenable)
                init_val = params_to_learn[name]
                setattr(self, name, nn.Parameter(torch.tensor([init_val], dtype=torch.float32)))
            else:
                # Si és fix (o no s'ha definit i posem un default), el registrem com a buffer (no entrenable)
                val = fixed_params.get(name, 1.0) # 1.0 per defecte si falta
                self.register_buffer(name, torch.tensor([val], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)