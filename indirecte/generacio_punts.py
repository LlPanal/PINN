import numpy as np
import os

print("Iniciant la simulació numèrica de Meinhardt...")

# Paràmetres del Model
alpha = 2.0
beta = 1.0
mu = 1.0
d1 = 1.0
d2 = 12.0
L = 25.0
t_max = 5.0

# Paràmetres de la Graella Numèrica
N = 100  
dx = L / (N - 1)
dy = L / (N - 1)
dt = 0.0001 
steps = int(t_max / dt)

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

#  Condició Inicial
u1_eq = beta / mu
u2_eq = (mu**2) / (alpha * beta)
n, m = 1.0, 5.0
pertorbacio = 0.1 * np.cos(n * np.pi * X / L) * np.cos(m * np.pi * Y / L)

u1 = u1_eq + pertorbacio
u2 = u2_eq + pertorbacio


# Llista per guardar els "snapshots" 
instants_a_guardar = np.linspace(0, t_max, 51) # Guarda dades cada 0.1 segons
dades_guardades = []
idx_guardat = 0

t = 0.0

for step in range(steps):
    if idx_guardat < len(instants_a_guardar) and t >= instants_a_guardar[idx_guardat]:
        print(f"Guardant snapshot a t = {t:.2f}")
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.full_like(x_flat, t)
        u1_flat = u1.flatten()
        u2_flat = u2.flatten()

        snapshot_data = np.column_stack((x_flat, y_flat, t_flat, u1_flat, u2_flat))
        dades_guardades.append(snapshot_data)
        idx_guardat += 1


    lap_u1 = (u1[:-2, 1:-1] + u1[2:, 1:-1] + u1[1:-1, :-2] + u1[1:-1, 2:] - 4*u1[1:-1, 1:-1]) / (dx**2)
    lap_u2 = (u2[:-2, 1:-1] + u2[2:, 1:-1] + u2[1:-1, :-2] + u2[1:-1, 2:] - 4*u2[1:-1, 1:-1]) / (dx**2)


    u1_in = u1[1:-1, 1:-1]
    u2_in = u2[1:-1, 1:-1]
    react_u1 = alpha * (u1_in**2) * u2_in - mu * u1_in
    react_u2 = beta - alpha * (u1_in**2) * u2_in

    u1[1:-1, 1:-1] += dt * (d1 * lap_u1 + react_u1)
    u2[1:-1, 1:-1] += dt * (d2 * lap_u2 + react_u2)

    # Condicions de Frontera de Neumann (Derivada zero a les vores)
    # Copiem els valors de la fila/columna adjacent per forçar que el gradient sigui nul
    # Eix Y (A dalt i a baix)
    u1[0, :] = u1[1, :]
    u1[-1, :] = u1[-2, :]
    
    # Eix X (Esquerra i dreta)
    u1[:, 0] = u1[:, 1]
    u1[:, -1] = u1[:, -2]  

    # El mateix per u2
    u2[0, :] = u2[1, :]
    u2[-1, :] = u2[-2, :]
    u2[:, 0] = u2[:, 1]
    u2[:, -1] = u2[:, -2]  

    t += dt

dataset_final = np.vstack(dades_guardades)

fitxer_sortida = "dades_sintetiques_meinhardt.npy"
np.save(fitxer_sortida, dataset_final)
print(f"S'han generat {dataset_final.shape[0]} punts de dades en total.")
print(f"Arxiu desat com: {fitxer_sortida}")