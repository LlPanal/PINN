import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from model import PINN
from train import train

# --- Paràmetres ---
layers = 4
neurons = 20
a = 1
u0 = 0.1
t_max = 10  
n_total_points = 1000
lr = 1e-3
epoques_per_etapa = 10000
model_path = f"logistic_pinn_{layers}layers_{neurons}neurons.pth"
train_model = True

torch.manual_seed(20) 

t_train = torch.linspace(0, t_max, n_total_points).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, t_max, n_total_points//10).view(-1, 1)


class HardConstraintPINN(nn.Module):
    def __init__(self, model,u0):
        super(HardConstraintPINN, self).__init__()
        self.model = model
        self.u0 = u0
    def forward(self, t):
        return self.u0 + t * self.model(t)

def logistic_residual(t, u):
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    # u_t = a * u * (1 - u)
    return u_t - a* u * (1 - u) 

def analytical_solution(t, a, u0):
    return 1 / (1 + ((1 - u0) / u0) * np.exp(-a * t))


base_model = PINN(input_dim=1, layers=layers, neurons=neurons)
model = HardConstraintPINN(base_model, u0)
ic = [(torch.tensor([[0.0]]), torch.tensor([[u0]]))]

# --- Entrenament ---
if os.path.exists(model_path) and not train_model:
    print(f"Carregant model des de {model_path}...")
    model.load_state_dict(torch.load(model_path))
else:
    model = train(
        model, 
        logistic_residual, 
        ic, 
        t_train, 
        epochs=epoques_per_etapa, 
        lr=lr,
        weight_bc=1
    )
    torch.save(model.state_dict(), model_path)

# --- Validació ---
def validate_pinn(model, t_test, a, u0):
    model.eval()
    
    with torch.no_grad():
        u_pinn = model(t_test).numpy()
    
    t_plot = t_test.numpy()
    u_real = analytical_solution(t_plot, a, u0)
    
    mse_error = np.mean((u_pinn - u_real)**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, u_real, 'r', label='Solució Analítica', linewidth=2, alpha=0.7)
    plt.scatter(t_plot, u_pinn, color='blue', s=10, label='Predicció PINN (Test Points)')
    plt.title(f'Validació en punts de Test (MSE: {mse_error:.2e})')
    plt.xlabel('Temps (t)')
    plt.ylabel('Població (u)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Error quadràtic mig en dades no vistes: {mse_error:.6f}")

validate_pinn(model, t_test, a, u0)