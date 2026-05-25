import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from model import PINN
from train_2eq import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilitzant {device}")

alpha = 2.0
beta = 1.0
mu = 1.0
d1 = 1.0
d2 = 12.0
L = 25.0
t_max = 5.0
model_path = "meinhardt_pinn.pth"

u1_eq = beta / mu
u2_eq = (mu**2) / (alpha * beta)

layers = 6
neurons = 64
n_collocation = 15000
n_ic = 2000
n_bc = 2000
lr = 1e-3
epochs = 10000
train_model = False  

torch.manual_seed(42)

# Punts de Col·locació (x, y, t)
x_col = torch.empty(n_collocation, 1, device=device).uniform_(0.0, L)
y_col = torch.empty(n_collocation, 1, device=device).uniform_(0.0, L)
t_col = torch.empty(n_collocation, 1, device=device).uniform_(0.0, t_max)
collocation_pts = torch.cat([x_col, y_col, t_col], dim=1).requires_grad_(True)

# Condició Inicial (t=0)
x_ic = torch.empty(n_ic, 1, device=device).uniform_(0.0, L)
y_ic = torch.empty(n_ic, 1, device=device).uniform_(0.0, L)
t_ic = torch.zeros(n_ic, 1, device=device)
xt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)

# Pertorbació 
n, m = 2.0, 5.0
pertorbacio = 0.1 * torch.cos(n * np.pi * x_ic / L) * torch.cos(m * np.pi * y_ic / L)
u1_target = torch.full((n_ic, 1), u1_eq, device=device) + pertorbacio
u2_target = torch.full((n_ic, 1), u2_eq, device=device) + pertorbacio

# Punts de Frontera
t_bc = torch.empty(n_bc, 1, device=device).uniform_(0.0, t_max)
y_bc = torch.empty(n_bc, 1, device=device).uniform_(0.0, L)
x_bc = torch.empty(n_bc, 1, device=device).uniform_(0.0, L)

x0_bc = torch.cat([torch.zeros(n_bc, 1, device=device), y_bc, t_bc], dim=1).requires_grad_(True)
xL_bc = torch.cat([torch.full((n_bc, 1), L, device=device), y_bc, t_bc], dim=1).requires_grad_(True)
y0_bc = torch.cat([x_bc, torch.zeros(n_bc, 1, device=device), t_bc], dim=1).requires_grad_(True)
yL_bc = torch.cat([x_bc, torch.full((n_bc, 1), L, device=device), t_bc], dim=1).requires_grad_(True)

bc_pts_dict = {
    'x': [x0_bc, xL_bc],
    'y': [y0_bc, yL_bc]
}

# Funcions de Pèrdua  
def initial_loss_fn(model):
    u_pred = model(xt_ic)
    loss_u1 = torch.mean((u_pred[:, 0:1] - u1_target)**2)
    loss_u2 = torch.mean((u_pred[:, 1:2] - u2_target)**2)
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

    f1 = u1_t - (alpha * (u1**2) * u2 - mu * u1 + d1 * (u1_xx + u1_yy))
    f2 = u2_t - (beta - alpha * (u1**2) * u2 + d2 * (u2_xx + u2_yy))

    return torch.mean(f1**2) + torch.mean(f2**2)

def avalua_i_visualitza(model, L, t_val, nx=100, ny=100):
    model.eval()

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    x_flat = X.flatten()[:, None]
    y_flat = Y.flatten()[:, None]
    t_flat = np.full_like(x_flat, t_val)

    x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device)
    xyt_tensor = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)

    with torch.no_grad():
        u_pred = model(xyt_tensor).cpu().numpy()

    u1_pred = u_pred[:, 0].reshape(ny, nx)
    u2_pred = u_pred[:, 1].reshape(ny, nx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f" t = {t_val}", fontsize=16)

    im1 = axes[0].pcolormesh(X, Y, u1_pred, cmap='viridis', shading='auto')
    axes[0].set_title("$u_1$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0], label="Concentració")

    im2 = axes[1].pcolormesh(X, Y, u2_pred, cmap='magma', shading='auto')
    axes[1].set_title("$u_2$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im2, ax=axes[1], label="Concentració")

    plt.tight_layout()
    plt.show()

# Entrenament i visualització
if __name__ == "__main__":
    model = PINN(input_dim=3, layers=layers, neurons=neurons, output_dim=2).to(device)
    if train_model or not os.path.exists(model_path):
        model = train(
            model=model,
            physics_loss_fn=physics_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            initial_loss_fn=initial_loss_fn,
            collocation_pts=collocation_pts,
            epochs=epochs,
            lr=lr,
            weight_ic=100.0,
            weight_bc=10.0
        )
        torch.save(model.state_dict(), "meinhardt_pinn.pth")
        print("Model entrenat i desat correctament.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model carregat")


    avalua_i_visualitza(model, L, t_val=0.0)
    avalua_i_visualitza(model, L, t_val=t_max/2.0)
    avalua_i_visualitza(model, L, t_val=t_max)