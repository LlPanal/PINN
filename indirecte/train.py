import torch

def train_invers_complet(
    model, 
    physics_loss_fn, 
    data_loss_fn,
    boundary_loss_fn,
    initial_loss_fn,
    collocation_pts, 
    epochs=15000, 
    lr=1e-3
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    w_data = 100.0
    w_ic = 100.0
    w_bc = 10.0
    w_physics = 1.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_data = data_loss_fn(model)
        loss_physics = physics_loss_fn(model, collocation_pts)
        loss_bc = boundary_loss_fn(model)
        loss_ic = initial_loss_fn(model)

        total_loss = (w_data * loss_data) + \
                     (w_physics * loss_physics) + \
                     (w_bc * loss_bc) + \
                     (w_ic * loss_ic)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print("-" * 60)
            print(f"Epoch {epoch+1:05d} | Pèrdua Total: {total_loss.item():.4f}")
            print(f"Errors: Data={loss_data.item():.8f}, Phys={loss_physics.item():.8f}, BC={loss_bc.item():.8f}, IC={loss_ic.item():.8f}")
            
            noms_fisics = ['alpha', 'beta', 'mu', 'd1', 'd2']
            params_str = ", ".join([
                f"{name}={param.item():.4f}" 
                for name, param in model.named_parameters() 
                if param.requires_grad and name in noms_fisics
            ])
            
            if params_str:
                print(f"Params: {params_str}\n")
            else:
                print("  -> (Model directe, cap paràmetre s'està aprenent)\n")

    return model