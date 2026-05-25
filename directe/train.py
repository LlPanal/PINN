import torch

def train(
    model, 
    physics_loss_fn, 
    ic_bc_points, 
    collocation_pts, 
    epochs=10000, 
    lr=1e-3,
    weight_bc=100.0
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_boundary = 0
        
        for x_cond, u_cond in ic_bc_points:
            
            u_pred = model(x_cond)
            loss_boundary += torch.mean((u_pred - u_cond)**2)

        u_pde = model(collocation_pts)
        t_col = collocation_pts[:, -1:]
        
        residu = physics_loss_fn(collocation_pts, u_pde)
        loss_physics = torch.mean( (residu**2))

        # Pèrdua total
        total_loss = weight_bc * loss_boundary + loss_physics
        
        total_loss.backward()
        optimizer.step()
        
        if epoch == 0 or (epoch +1) % 200 == 0:
            print(f"  Epoch {epoch+1} | Loss BC: {loss_boundary.item():.8f} | Loss Phys: {loss_physics.item():.8f}")

    return model
