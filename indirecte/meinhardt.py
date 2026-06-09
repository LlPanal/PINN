import torch
import numpy as np
import os

from model import PINN
from train import train_invers_complet 

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Utilitzant dispositiu: {device}")

L = 25.0
t_max = 5.0
data_path = "dades_sintetiques_meinhardt.npy"
model_path = "meinhardt_pinn_inverse.pth"
epochs = 50000
lr = 1e-3
n, m = 5.0, 0.0
n_data = 10000
n_collocation = 20000
n_ic = 2000
n_bc = 2000
w_data = 100.0
w_ic = 100.0
w_bc = 10.0
w_physics = 1.0


torch.manual_seed(42)

"""
Valors reals del model de Meinhardt:
alpha = 2.0
beta = 1.0
mu = 1.0
d1 = 1.0
d2 = 12.0
"""
parametres_a_aprendre = {
    'alpha': 5,  
    'beta': 0.5  

}
parametres_fixos = {
    'mu': 2.0,
    'd1': 0.1,  
    'd2': 5.0
}


# Preparació de les dades
dataset = np.load(data_path)
temps_permesos = np.linspace(0, t_max, 11)
mask_train = np.isin(np.round(dataset[:, 2], decimals=4), np.round(temps_permesos, decimals=4))
dataset= dataset[mask_train]
idx_aleatoris = np.random.choice(dataset.shape[0], n_data, replace=False)
dades_entrenament = dataset[idx_aleatoris, :]
xyt_data = torch.tensor(dades_entrenament[:, 0:3], dtype=torch.float32, device=device)
u_target = torch.tensor(dades_entrenament[:, 3:5], dtype=torch.float32, device=device)


# punts de colocaicio, condicio inicial i frontera
x_col = torch.empty(n_collocation, 1, device=device).uniform_(0.0, L)
y_col = torch.empty(n_collocation, 1, device=device).uniform_(0.0, L)
t_col = torch.empty(n_collocation, 1, device=device).uniform_(0.0, t_max)
collocation_pts = torch.cat([x_col, y_col, t_col], dim=1).requires_grad_(True)

x_ic = torch.empty(n_ic, 1, device=device).uniform_(0.0, L)
y_ic = torch.empty(n_ic, 1, device=device).uniform_(0.0, L)
t_ic = torch.zeros(n_ic, 1, device=device)
xt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)

u1_eq = parametres_fixos.get('beta', 1.0) / parametres_fixos.get('mu', 1.0)
u2_eq = (parametres_fixos.get('mu', 1.0)**2) / (parametres_fixos.get('alpha', 2.0) * parametres_fixos.get('beta', 1.0))
pertorbacio = 0.1 * torch.cos(n * np.pi * x_ic / L) * torch.cos(m * np.pi * y_ic / L)
u1_ic_target = torch.full((n_ic, 1), u1_eq, device=device) + pertorbacio
u2_ic_target = torch.full((n_ic, 1), u2_eq, device=device) + pertorbacio

t_bc = torch.empty(n_bc, 1, device=device).uniform_(0.0, t_max)
y_bc = torch.empty(n_bc, 1, device=device).uniform_(0.0, L)
x_bc = torch.empty(n_bc, 1, device=device).uniform_(0.0, L)
x0_bc = torch.cat([torch.zeros(n_bc, 1, device=device), y_bc, t_bc], dim=1).requires_grad_(True)
xL_bc = torch.cat([torch.full((n_bc, 1), L, device=device), y_bc, t_bc], dim=1).requires_grad_(True)
y0_bc = torch.cat([x_bc, torch.zeros(n_bc, 1, device=device), t_bc], dim=1).requires_grad_(True)
yL_bc = torch.cat([x_bc, torch.full((n_bc, 1), L, device=device), t_bc], dim=1).requires_grad_(True)
bc_pts_dict = {'x': [x0_bc, xL_bc], 'y': [y0_bc, yL_bc]}


# Funcions de perdua
def data_loss_fn(model):
    u_pred = model(xyt_data)
    return torch.mean((u_pred - u_target)**2)

def initial_loss_fn(model):
    u_pred = model(xt_ic)
    loss_u1 = torch.mean((u_pred[:, 0:1] - u1_ic_target)**2)
    loss_u2 = torch.mean((u_pred[:, 1:2] - u2_ic_target)**2)
    return loss_u1 + loss_u2

def boundary_loss_fn(model):
    loss = 0
    for axis, pt_list in bc_pts_dict.items():
        idx = 0 if axis == 'x' else 1
        for pts in pt_list:
            u_pred = model(pts)
            u1, u2 = u_pred[:, 0:1], u_pred[:, 1:2]
            grad_u1 = torch.autograd.grad(u1, pts, torch.ones_like(u1), create_graph=True)[0]
            grad_u2 = torch.autograd.grad(u2, pts, torch.ones_like(u2), create_graph=True)[0]
            loss += torch.mean(grad_u1[:, idx:idx+1]**2) + torch.mean(grad_u2[:, idx:idx+1]**2)
    return loss

def physics_loss_fn(model, collocation_pts):
    u_pred = model(collocation_pts)
    u1, u2 = u_pred[:, 0:1], u_pred[:, 1:2]

    g1 = torch.autograd.grad(u1, collocation_pts, torch.ones_like(u1), create_graph=True)[0]
    u1_x, u1_y, u1_t = g1[:, 0:1], g1[:, 1:2], g1[:, 2:3]
    u1_xx = torch.autograd.grad(u1_x, collocation_pts, torch.ones_like(u1_x), create_graph=True)[0][:, 0:1]
    u1_yy = torch.autograd.grad(u1_y, collocation_pts, torch.ones_like(u1_y), create_graph=True)[0][:, 1:2]

    g2 = torch.autograd.grad(u2, collocation_pts, torch.ones_like(u2), create_graph=True)[0]
    u2_x, u2_y, u2_t = g2[:, 0:1], g2[:, 1:2], g2[:, 2:3]
    u2_xx = torch.autograd.grad(u2_x, collocation_pts, torch.ones_like(u2_x), create_graph=True)[0][:, 0:1]
    u2_yy = torch.autograd.grad(u2_y, collocation_pts, torch.ones_like(u2_y), create_graph=True)[0][:, 1:2]

    f1 = u1_t - (model.alpha * (u1**2) * u2 - model.mu * u1 + model.d1 * (u1_xx + u1_yy))
    f2 = u2_t - (model.beta - model.alpha * (u1**2) * u2 + model.d2 * (u2_xx + u2_yy))

    return torch.mean(f1**2) + torch.mean(f2**2)

#Entrenament
if __name__ == "__main__":
    model = PINN(
        input_dim=3, layers=6, neurons=64, output_dim=2,
        params_to_learn=parametres_a_aprendre, 
        fixed_params=parametres_fixos
    ).to(device)
    print("\nIniciant l'entrenament de la PINN (Modular)...")
    print(f"Fixats: {parametres_fixos}")
    print(f"A Aprendre: {parametres_a_aprendre}\n")
    
    model = train_invers_complet(
        model=model,
        physics_loss_fn=physics_loss_fn,
        data_loss_fn=data_loss_fn,
        boundary_loss_fn=boundary_loss_fn,
        initial_loss_fn=initial_loss_fn,
        collocation_pts=collocation_pts,
        epochs=epochs, 
        lr=lr,
        w_data=w_data,
        w_ic=w_ic,
        w_bc=w_bc,
        w_physics=w_physics
    )

    torch.save(model.state_dict(), model_path)
    print("\nModel desat")