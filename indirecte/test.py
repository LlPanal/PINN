import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilitzant dispositiu: {device}")

L = 25.0
nx, ny = 100, 100  
data_path = "dades_sintetiques_meinhardt.npy"
model_path = "meinhardt_pinn_inverse_a_b.pth"
batch_size = 20000  

try:
    dataset = np.load(data_path)
except FileNotFoundError:
    raise FileNotFoundError("No s'ha trobat l'arxiu de dades (.npy). Genera'l primer.")

n_punts_totals = dataset.shape[0]
X_data = dataset[:, 0:3]  # Entrades: (x, y, t)
Y_true = dataset[:, 3:5]  # Sortides reals: (u1, u2)

model = PINN(input_dim=3, layers=6, neurons=64, output_dim=2).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model de la PINN carregat correctament.")
except FileNotFoundError:
    raise FileNotFoundError(f"No s'ha trobat el model {model_path}.")


preds_list = []

with torch.no_grad():
    for i in range(0, n_punts_totals, batch_size):
        X_batch = X_data[i : i + batch_size]
        x_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
        y_pred_batch = model(x_tensor).cpu().numpy()
        preds_list.append(y_pred_batch)

Y_pred = np.vstack(preds_list)

mse_u1 = np.mean((Y_true[:, 0] - Y_pred[:, 0])**2)
mse_u2 = np.mean((Y_true[:, 1] - Y_pred[:, 1])**2)

print("\n================ RESULTATS DE L'AVALUACIÓ GLOBAL ================")
print(f"u1 -> MSE: {mse_u1:.6e}")
print(f"u2 -> MSE: {mse_u2:.6e}")
print("=================================================================")
print(f"Alpha : {model.alpha.item():.4f} (Real: 2.0)")
print(f"Beta  : {model.beta.item():.4f} (Real: 1.0)")
print(f"Mu    : {model.mu.item():.4f} (Real: 1.0)")
print(f"D1    : {model.d1.item():.4f} (Real: 1.0)")
print(f"D2    : {model.d2.item():.4f} (Real: 12.0)")
print("=================================================================\n")



time_steps = [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]

for t_step in time_steps:
    mask = np.isclose(dataset[:, 2], t_step, atol=1e-4)
    data_t = dataset[mask]
    
    if len(data_t) == 0:
        print(f"No s'han trobat dades  per a t={t_step}")
        continue
    
    X = data_t[:, 0].reshape(ny, nx)
    Y = data_t[:, 1].reshape(ny, nx)
    u1_num = data_t[:, 3].reshape(ny, nx)
    u2_num = data_t[:, 4].reshape(ny, nx)
    
    x_flat = X.flatten()[:, None]
    y_flat = Y.flatten()[:, None]
    t_flat = np.full_like(x_flat, t_step)
    
    x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device)
    xyt_tensor = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
    
    with torch.no_grad():
        u_pred = model(xyt_tensor).cpu().numpy()
        
    u1_pinn = u_pred[:, 0].reshape(ny, nx)
    u2_pinn = u_pred[:, 1].reshape(ny, nx)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Comparativa PINN Inversa vs Dada Sintètica a t = {t_step}", fontsize=18, fontweight='bold')
    
    im00 = axes[0, 0].pcolormesh(X, Y, u1_num, cmap='viridis', shading='auto')
    axes[0, 0].set_title("Dada numèrica ($u_1$)")
    fig.colorbar(im00, ax=axes[0, 0])
    
    im01 = axes[0, 1].pcolormesh(X, Y, u1_pinn, cmap='viridis', shading='auto')
    axes[0, 1].set_title("Predicció PINN ($u_1$)")
    fig.colorbar(im01, ax=axes[0, 1])
    
    err_u1 = np.abs(u1_num - u1_pinn)
    im02 = axes[0, 2].pcolormesh(X, Y, err_u1, cmap='Reds', shading='auto')
    axes[0, 2].set_title(f"Error absolut ($u_1$) | Màx: {np.max(err_u1):.4f}")
    fig.colorbar(im02, ax=axes[0, 2])
    
    im10 = axes[1, 0].pcolormesh(X, Y, u2_num, cmap='magma', shading='auto')
    axes[1, 0].set_title("Dada numèrica ($u_2$)")
    fig.colorbar(im10, ax=axes[1, 0])
    
    im11 = axes[1, 1].pcolormesh(X, Y, u2_pinn, cmap='magma', shading='auto')
    axes[1, 1].set_title("Predicció PINN ($u_2$)")
    fig.colorbar(im11, ax=axes[1, 1])
    
    err_u2 = np.abs(u2_num - u2_pinn)
    im12 = axes[1, 2].pcolormesh(X, Y, err_u2, cmap='Reds', shading='auto')
    axes[1, 2].set_title(f"Error absolut ($u_2$) | Màx: {np.max(err_u2):.4f}")
    fig.colorbar(im12, ax=axes[1, 2])
    
    for ax in axes.flatten():
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    plt.tight_layout()
    plt.show()
