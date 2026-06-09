import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erfcx
import os
from model import PINN
from train import train

# --- Paràmetres ---
layers = 6                
neurons = 40
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0 , 1.0   
n_collocation = 5000     
n_bc = 500                
n_ic = 500                
lr = 1e-3

nu = 0.01 / np.pi 

epochs = 10000
model_path = f"burgers_pinn_{layers}layers_{neurons}neurons.pth"
train_model = False

torch.manual_seed(42)

#punts del mig, per la pèrdua física
x_col = torch.empty(n_collocation, 1).uniform_(x_min, x_max)
t_col = torch.empty(n_collocation, 1).uniform_(0.0, t_max)
xt_col = torch.cat([x_col, t_col], dim=1).requires_grad_(True)

# Condicions Inicials: u(x,0) = -1 si x<0, 1 si x>0 (suavitzat per autograd)
x_ic = torch.empty(n_ic, 1).uniform_(x_min, x_max)
t_ic = torch.zeros(n_ic, 1)
xt_ic = torch.cat([x_ic, t_ic], dim=1)
u_ic = torch.where(x_ic < 0, torch.zeros_like(x_ic) - 1.0, torch.zeros_like(x_ic) + 1.0)

# Condicions de Frontera: u(-1,t) = -1 i u(1,t) = 1
t_bc = torch.empty(n_bc, 1).uniform_(0.0, t_max)

x_bc_left = torch.full((n_bc, 1), x_min)
xt_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
u_bc_left = torch.zeros(n_bc, 1) - 1.0

x_bc_right = torch.full((n_bc, 1), x_max)
xt_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
u_bc_right = torch.zeros(n_bc, 1) + 1.0

ic_bc_points = [
    (xt_ic, u_ic),
    (xt_bc_left, u_bc_left),
    (xt_bc_right, u_bc_right)
]

# --- Funció de Pèrdua Física ---
def burgers_residual(xt, u):
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1] 
    u_t = grads[:, 1:2] 
    
    # u_t + u*u_x = 0  
    return u_t + u * u_x 


# --- Solució Real ---
import numpy as np

def exact_solution_inviscid(x_grid, t_grid):
    T, X = np.meshgrid(t_grid, x_grid, indexing='ij')
    
    U = np.zeros_like(T)
    
    # A t=0, apliquem la condició inicial (esglaó)
    U[0, :] = np.where(x_grid < 0, -1.0, 1.0)
    
    # Per t > 0, separem les variables
    t_pos = T[1:, :]
    x_pos = X[1:, :]
    
    # Definim les 3 zones de l'ona de rarefacció per a t > 0
    condicions = [
        x_pos <= -t_pos,                      # Zona esquerra plana
        x_pos >= t_pos,                       # Zona dreta plana
        (x_pos > -t_pos) & (x_pos < t_pos)    # Zona central (la rampa)
    ]
    
    # Els valors de u(x,t) que corresponen a cadascuna de les 3 zones
    valors = [
        -1.0,
        1.0,
        x_pos / t_pos
    ]
    
    # np.select aplica les condicions de manera vectoritzada i molt eficient
    U[1:, :] = np.select(condicions, valors)
    
    return U
    

model = PINN(input_dim=2, layers=layers, neurons=neurons)


# --- Entrenament ---
if os.path.exists(model_path) and not train_model:
    print(f"Carregant model des de {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    model = train(
        model, 
        burgers_residual, 
        ic_bc_points, 
        xt_col, 
        epochs=epochs, 
        lr=lr,
        weight_bc=100.0 
    )
    torch.save(model.state_dict(), model_path)

# --- Validació ---
def plot_burgers(model):
    model.eval()
    
    x_grid = np.linspace(x_min, x_max, 1000)
    t_grid = np.linspace(0, t_max, 100)
    X, T = np.meshgrid(x_grid, t_grid)
    
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    XT_tensor = torch.tensor(np.hstack((X_flat, T_flat)), dtype=torch.float32)
    
    with torch.no_grad():
        U_pred = model(XT_tensor).numpy().reshape(X.shape)
        
    # solució real
    U_real_matrix = exact_solution_inviscid(x_grid, t_grid)
    

    mse_total = np.mean((U_real_matrix - U_pred)**2)
    print(f"\n" + "="*30)
    print(f"MSE Total del Model: {mse_total:.6e}")
    print("="*30 + "\n")
        

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(T, X, U_pred, cmap='jet', shading='auto')
    plt.colorbar(label='Amplitud $u(x, t)$')
    plt.title(f"Solució PINN (MSE: {mse_total:.2e})")
    plt.xlabel('Temps ($t$)')
    plt.ylabel('Posició ($x$)')
    plt.show()

    # Visualització de talls temporals i càlcul de MSE local
    plt.figure(figsize=(12, 7))
    times_to_plot = [0.0, 0.20, 0.50, 0.80]
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, t_val in enumerate(times_to_plot):
        idx_t = np.abs(t_grid - t_val).argmin()
        
        u_pinn = U_pred[idx_t, :]
        u_real = U_real_matrix[idx_t, :]
        
        
        plt.plot(x_grid, u_real, color=colors[idx], linestyle='-', alpha=0.3, linewidth=4, label=f'Real t={t_val}')
        plt.plot(x_grid, u_pinn, color=colors[idx], linestyle='--', linewidth=2, label=f'PINN')

    plt.title('Comparativa PINN vs Solució Exacta')
    plt.xlabel('Posició ($x$)')
    plt.ylabel('Amplitud $u(x, t)$')
    plt.legend(loc='best', fontsize='x-large')    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_burgers(model)