import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from model import PINN
from train import train

# --- Paràmetres ---
d = 1.0   # Coeficient de difusió
a = 1.0    # Taxa de creixement
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

# --- Paràmetres d'entrenament ---
layers = 6
neurons = 40
n_collocation = 5000
n_ic = 500
n_bc = 500
lr = 1e-3
epochs = 25000
model_path = f"fisher_pinn_{layers}layers_{neurons}neurons.pth"
train_model = False

torch.manual_seed(42)

# 1. Punts de col·locació (Física)
x_col = torch.empty(n_collocation, 1).uniform_(x_min, x_max)
t_col = torch.empty(n_collocation, 1).uniform_(t_min, t_max)
xt_col = torch.cat([x_col, t_col], dim=1).requires_grad_(True)

# 2. Condició Inicial (IC): Funció d'Heaviside (u=1 si x <= 0, u=0 si x > 0)
x_ic = torch.empty(n_ic, 1).uniform_(x_min, x_max)
t_ic = torch.zeros(n_ic, 1)
xt_ic = torch.cat([x_ic, t_ic], dim=1)
u_ic = torch.where(x_ic <= 0, torch.ones_like(x_ic), torch.zeros_like(x_ic))

# 3. Condicions de Frontera (BC)
t_bc = torch.empty(n_bc, 1).uniform_(t_min, t_max)

# Frontera esquerra (x = -1): u = 1 (coherent amb Heaviside per x <= 0)
x_bc_left = torch.full((n_bc, 1), x_min)
xt_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
u_bc_left = torch.ones(n_bc, 1) 

# Frontera dreta (x = 1): u = 0
x_bc_right = torch.full((n_bc, 1), x_max)
xt_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
u_bc_right = torch.zeros(n_bc, 1)

ic_bc_points = [
    (xt_ic, u_ic),
    (xt_bc_left, u_bc_left),
    (xt_bc_right, u_bc_right)
]

# --- Funció de Pèrdua Física ---
def fisher_kpp_residual(xt, u):
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    grads_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grads_xx[:, 0:1]
    
    return u_t - d * u_xx - a * u * (1 - u)

# --- Solució Real (Simulador Numèric) ---
def exact_solution_fisher(x_grid, t_grid, d, a):
    dx = x_grid[1] - x_grid[0]
    
    def fisher_rhs(t, u):
        dudt = np.zeros_like(u)
        u_interior = u[1:-1]
        u_esquerra = u[:-2]
        u_dreta = u[2:]
        
        diffusion = d * (u_dreta - 2*u_interior + u_esquerra) / (dx**2)
        reaction = a * u_interior * (1 - u_interior)
        
        dudt[1:-1] = diffusion + reaction
        
        # Fronteres: Mantinguem u(-1)=1 i u(1)=0
        dudt[0] = 0
        dudt[-1] = 0
        return dudt
        
    # IC d'Heaviside per al simulador
    u0 = np.where(x_grid <= 0, 1.0, 0.0)
    
    sol = solve_ivp(fisher_rhs, [t_grid[0], t_grid[-1]], u0, t_eval=t_grid, method='BDF')
    return sol.y.T

# --- Inicialització i Entrenament ---
model = PINN(input_dim=2, layers=layers, neurons=neurons)

if os.path.exists(model_path) and not train_model:
    print(f"Carregant model des de {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    print("Iniciant entrenament")
    model = train(
        model, 
        fisher_kpp_residual, 
        ic_bc_points, 
        xt_col, 
        epochs=epochs, 
        lr=lr,
        weight_bc=100.0
    )
    torch.save(model.state_dict(), model_path)

# --- Validació i Comparació ---
def plot_results(model):
    model.eval()
    
    x_grid = np.linspace(x_min, x_max, 512)
    t_grid = np.linspace(t_min, t_max, 100)
    X, T = np.meshgrid(x_grid, t_grid)
    
    xt_flat = torch.tensor(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), dtype=torch.float32)
    with torch.no_grad():
        U_pred = model(xt_flat).numpy().reshape(X.shape)
        
    U_real_matrix = exact_solution_fisher(x_grid, t_grid, d, a)
    
    # Gràfica 1: Heatmap 2D
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(T, X, U_pred, cmap='jet', shading='auto')
    plt.colorbar(label='u(x, t)')
    plt.title('PINN Fisher: Evolució de la funció Heaviside')
    plt.xlabel('Temps (t)')
    plt.ylabel('Espai (x)')
    plt.show()

    # Gràfica 2: Talls temporals
    plt.figure(figsize=(12, 7))
    times_to_plot = [0.0, 0.1, 0.25, 0.50, 0.99]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for idx, t_val in enumerate(times_to_plot):
        idx_t = np.abs(t_grid - t_val).argmin()
        u_pinn = U_pred[idx_t, :]
        u_real = U_real_matrix[idx_t, :]
        
        plt.plot(x_grid, u_real, color=colors[idx], linestyle='-', alpha=0.3, linewidth=4, label=f'BDF t={t_val}')
        plt.plot(x_grid, u_pinn, color=colors[idx], linestyle='--', linewidth=2, label=f'PINN t={t_val}')

    plt.title('Comparació: Front d\'ona des de condició d\'Heaviside')
    plt.xlabel('Posició (x)')
    plt.ylabel('u(x, t)')
    plt.legend(loc='best', fontsize='x-large')    
    plt.grid(True, alpha=0.3)
    plt.show()

plot_results(model)