import torch

def train(
    model, 
    physics_loss_fn, 
    boundary_loss_fn,
    initial_loss_fn,
    collocation_pts, 
    epochs=10000, 
    lr=1e-3,
    weight_ic=100.0,
    weight_bc=10.0
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_ic = initial_loss_fn(model)
        loss_bc = boundary_loss_fn(model)
        loss_physics = physics_loss_fn(model, collocation_pts)

        total_loss = weight_ic * loss_ic + weight_bc * loss_bc + loss_physics
        
        total_loss.backward()
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:05d} | Total: {total_loss.item():.8f} | IC: {loss_ic.item():.8f} | BC: {loss_bc.item():.8f} | Phys: {loss_physics.item():.8f}")

    return model