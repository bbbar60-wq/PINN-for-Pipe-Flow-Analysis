# PINN_Phase2_Pipe_Flow.ipynb

# 1. Library Imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Set default tensor type to float32 for consistency
torch.set_default_dtype(torch.float32)

# Use a GPU if available, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 2. The PINN Model Architecture
class PINN(nn.Module):
    """
    Defines the neural network architecture for the PINN.
    Input: (r, z) coordinates
    Output: (vr, vz, p) - radial velocity, axial velocity, pressure
    """

    def __init__(self, layers):
        super(PINN, self).__init__()

        # Build the neural network layers
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"activation_{i}", nn.Tanh())

    def forward(self, x):
        """Forward pass: maps input tensor x to the output."""
        return self.net(x)


# 3. Physics-Based Loss Function
def pde_loss(model, r, z, rho, mu):\
    """
    Computes the loss based on the Navier-Stokes and continuity equations.
    This function uses automatic differentiation to get the required derivatives.
    """
    # Clone and set requires_grad=True to compute derivatives
    r = r.clone().requires_grad_(True)
    z = z.clone().requires_grad_(True)

    # Get model predictions
    predictions = model(torch.cat([r, z], dim=1))
    vr, vz, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

    # Compute first-order derivatives using torch.autograd.grad
    dv_r_dr = torch.autograd.grad(vr.sum(), r, create_graph=True)[0]
    dv_r_dz = torch.autograd.grad(vr.sum(), z, create_graph=True)[0]

    dv_z_dr = torch.autograd.grad(vz.sum(), r, create_graph=True)[0]
    dv_z_dz = torch.autograd.grad(vz.sum(), z, create_graph=True)[0]

    dp_dr = torch.autograd.grad(p.sum(), r, create_graph=True)[0]
    dp_dz = torch.autograd.grad(p.sum(), z, create_graph=True)[0]

    # Compute second-order derivatives
    d2v_r_dr2 = torch.autograd.grad(dv_r_dr.sum(), r, create_graph=True)[0]
    d2v_r_dz2 = torch.autograd.grad(dv_r_dz.sum(), z, create_graph=True)[0]

    d2v_z_dr2 = torch.autograd.grad(dv_z_dr.sum(), r, create_graph=True)[0]
    d2v_z_dz2 = torch.autograd.grad(dv_z_dz.sum(), z, create_graph=True)[0]

    # PDE Residuals for axisymmetric, steady-state, incompressible flow

    # Continuity equation residual (div(v) = 0)
    # Note: Adding a small epsilon to r to avoid division by zero at the centerline (r=0)
    epsilon = 1e-8
    loss_continuity = dv_r_dr + vr / (r + epsilon) + dv_z_dz

    # r-Momentum equation residual
    loss_r_momentum = (rho * (vr * dv_r_dr + vz * dv_r_dz) -
                       (-dp_dr + mu * (
                               d2v_r_dr2 + (1 / (r + epsilon)) * dv_r_dr - vr / (r + epsilon) ** 2 + d2v_r_dz2)))

    # z-Momentum equation residual
    loss_z_momentum = (rho * (vr * dv_z_dr + vz * dv_z_dz) -
                       (-dp_dz + mu * (d2v_z_dr2 + (1 / (r + epsilon)) * dv_z_dr + d2v_z_dz2)))

    # Calculate the mean squared error for each residual
    pde_loss_vr = torch.mean(loss_r_momentum ** 2)
    pde_loss_vz = torch.mean(loss_z_momentum ** 2)
    pde_loss_c = torch.mean(loss_continuity ** 2)

    return pde_loss_vr + pde_loss_vz + pde_loss_c


# 4. Problem Setup & Training Data

# Fluid and Geometry Parameters
R = 1.0  # Pipe radius
L = 5.0  # Pipe length
rho = 1.0  # Fluid density
mu = 0.02  # Fluid viscosity
Pin = 5.0  # Inlet pressure (relative to outlet)
Pout = 0.0  # Outlet pressure

# Number of training points
N_pde = 5000  # Collocation points inside the domain
N_bc = 1000  # Boundary points on each boundary

# Define the network architecture: [input_dim, hidden_layers..., output_dim]
layers = [2, 30, 30, 30, 30, 30, 3]  # 2 inputs (r,z), 5 hidden layers of 30 neurons, 3 outputs (vr, vz, p)
pinn = PINN(layers).to(device)

# --- Generate Training Points ---

# PDE (collocation) points (randomly sampled inside the domain)
r_pde = torch.sqrt(torch.rand(N_pde, 1)) * R  # Use sqrt for uniform area sampling
z_pde = torch.rand(N_pde, 1) * L
r_pde, z_pde = r_pde.to(device), z_pde.to(device)

# Boundary Condition points
# Inlet (z=0)
r_inlet = torch.rand(N_bc, 1) * R
z_inlet = torch.zeros(N_bc, 1)
p_inlet_gt = torch.full((N_bc, 1), Pin)  # Ground truth pressure at inlet
r_inlet, z_inlet, p_inlet_gt = r_inlet.to(device), z_inlet.to(device), p_inlet_gt.to(device)

# Outlet (z=L)
r_outlet = torch.rand(N_bc, 1) * R
z_outlet = torch.full((N_bc, 1), L)
p_outlet_gt = torch.full((N_bc, 1), Pout)  # Ground truth pressure at outlet
r_outlet, z_outlet, p_outlet_gt = r_outlet.to(device), z_outlet.to(device), p_outlet_gt.to(device)

# Pipe Wall (r=R)
r_wall = torch.full((N_bc, 1), R)
z_wall = torch.rand(N_bc, 1) * L
vel_wall_gt = torch.zeros(N_bc, 2)  # Ground truth velocity (vr, vz) = (0,0) at wall
r_wall, z_wall, vel_wall_gt = r_wall.to(device), z_wall.to(device), vel_wall_gt.to(device)

# Centerline (r=0) - symmetry condition
r_center = torch.zeros(N_bc, 1)
z_center = torch.rand(N_bc, 1) * L
r_center, z_center = r_center.to(device), z_center.to(device)

# 5. Training the PINN

# Optimizer
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)  # Learning rate scheduler

epochs = 15000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    # --- Calculate Boundary Condition Losses ---

    # Inlet Pressure Loss (and no radial velocity)
    inlet_preds = pinn(torch.cat([r_inlet, z_inlet], dim=1))
    loss_inlet_p = torch.mean((inlet_preds[:, 2:3] - p_inlet_gt) ** 2)
    loss_inlet_vr = torch.mean(inlet_preds[:, 0:1] ** 2)  # vr should be 0

    # Outlet Pressure Loss (and no radial velocity)
    outlet_preds = pinn(torch.cat([r_outlet, z_outlet], dim=1))
    loss_outlet_p = torch.mean((outlet_preds[:, 2:3] - p_outlet_gt) ** 2)
    loss_outlet_vr = torch.mean(outlet_preds[:, 0:1] ** 2)  # vr should be 0

    # Wall No-Slip Loss (vr=0, vz=0)
    wall_preds = pinn(torch.cat([r_wall, z_wall], dim=1))
    loss_wall = torch.mean(wall_preds[:, 0:2] ** 2)

    # Centerline Symmetry Loss (vr=0, dvz/dr=0)
    r_center_c = r_center.clone().requires_grad_(True)
    center_preds = pinn(torch.cat([r_center_c, z_center], dim=1))
    dvz_dr_center = torch.autograd.grad(center_preds[:, 1:2].sum(), r_center_c, create_graph=True)[0]
    loss_center = torch.mean(center_preds[:, 0:1] ** 2) + torch.mean(dvz_dr_center ** 2)

    # Total Boundary Loss
    loss_bc = loss_inlet_p + loss_outlet_p + loss_wall + loss_center + loss_inlet_vr + loss_outlet_vr

    # --- Calculate PDE Loss ---
    loss_pde_val = pde_loss(pinn, r_pde, z_pde, rho, mu)

    # --- Total Loss (with weighting) ---
    # Weighting can help balance the influence of different loss components.
    # Here, we give the boundary conditions more weight.
    lambda_bc = 100.0
    total_loss = loss_pde_val + lambda_bc * loss_bc
    loss_history.append(total_loss.item())

    # Backpropagation and optimization step
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 1000 == 0:
        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4e}, BC Loss: {loss_bc.item():.4e}, PDE Loss: {loss_pde_val.item():.4e}')

# 6. Visualization
print("\n--- Training Finished. Visualizing Results. ---")

# Create a grid for plotting
r_range = np.linspace(0, R, 100)
z_range = np.linspace(0, L, 200)
r_grid, z_grid = np.meshgrid(r_range, z_range)
r_flat = r_grid.flatten()
z_flat = z_grid.flatten()

# Prepare grid for PyTorch
r_tensor = torch.tensor(r_flat, dtype=torch.float32).unsqueeze(1).to(device)
z_tensor = torch.tensor(z_flat, dtype=torch.float32).unsqueeze(1).to(device)

# Evaluate model on the grid
pinn.eval()
with torch.no_grad():
    predictions = pinn(torch.cat([r_tensor, z_tensor], dim=1)).cpu().numpy()

vr_pred = predictions[:, 0].reshape(r_grid.shape)
vz_pred = predictions[:, 1].reshape(z_grid.shape)
p_pred = predictions[:, 2].reshape(r_grid.shape)

# Analytical solution for fully developed flow (Hagen-Poiseuille)
dP_dz = (Pout - Pin) / L
r_analytical = torch.linspace(0, R, 100)
vz_analytical = - (1 / (4 * mu)) * dP_dz * (R ** 2 - r_analytical ** 2)

# --- ADD THIS BLOCK TO SAVE DATA ---

print("Saving numerical data to file...")
np.savez_compressed(
    "pinn_pipe_flow_data.npz",
    r_grid=r_grid,
    z_grid=z_grid,
    vr_predicted=vr_pred,
    vz_predicted=vz_pred,
    p_predicted=p_pred,
    vz_analytical=vz_analytical.numpy(), # Save the analytical solution for comparison
    loss_history=np.array(loss_history)
)
print("Data saved successfully.")

# --- Create Plots ---
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3)

# Axial Velocity Contour
ax1 = fig.add_subplot(gs[0, :])
c1 = ax1.contourf(z_grid, r_grid, vz_pred, levels=50, cmap='viridis')
fig.colorbar(c1, ax=ax1, label='Axial Velocity (vz)')
ax1.set_title('Predicted Axial Velocity (vz) Field')
ax1.set_xlabel('z (pipe length)')
ax1.set_ylabel('r (pipe radius)')
ax1.set_aspect('equal')

# Pressure Contour
ax2 = fig.add_subplot(gs[1, :])
c2 = ax2.contourf(z_grid, r_grid, p_pred, levels=50, cmap='plasma')
fig.colorbar(c2, ax=ax2, label='Pressure (p)')
ax2.set_title('Predicted Pressure (p) Field')
ax2.set_xlabel('z (pipe length)')
ax2.set_ylabel('r (pipe radius)')
ax2.set_aspect('equal')

# Velocity Profile Comparison
ax3 = fig.add_subplot(gs[2, 0])
# Get profile from the middle of the pipe for comparison
z_mid_index = int(len(z_range) / 2)
vz_profile_pinn = vz_pred[z_mid_index, :]
ax3.plot(r_range, vz_profile_pinn, 'r-', label='PINN Prediction', linewidth=2)
ax3.plot(r_analytical.numpy(), vz_analytical.numpy(), 'b--', label='Analytical Solution', linewidth=2)
ax3.set_title('Velocity Profile at z = L/2')
ax3.set_xlabel('r (pipe radius)')
ax3.set_ylabel('Axial Velocity (vz)')
ax3.legend()
ax3.grid(True)

# Loss History Plot
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(loss_history)
ax4.set_yscale('log')
ax4.set_title('Training Loss History')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Log Loss')
ax4.grid(True)
plt.savefig("PINN_Pipe_Flow_Results.png", dpi=300, bbox_inches='tight')
plt.show()