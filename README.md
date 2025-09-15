# Physics-Informed Neural Network (PINN) for Pipe Flow Analysis

This project demonstrates the application of a Physics-Informed Neural Network (PINN) to solve for the velocity profile of a laminar, incompressible fluid flow in a circular pipe. This problem is governed by a simplified form of the Navier-Stokes equations.

---

## 📜 Project Description

Traditional numerical methods like Finite Element or Finite Difference are commonly used to solve differential equations. This project offers an alternative, deep learning-based approach. A PINN is a neural network that is trained to solve supervised learning tasks while respecting the laws of physics described by general nonlinear partial differential equations.

Here, we model the steady, fully developed laminar flow in a pipe, which simplifies to a second-order ordinary differential equation (ODE) for the axial velocity profile, $u(r)$. The PINN learns the velocity profile by minimizing a loss function that includes both the governing physical equation and the boundary conditions.

---

## ✨ Key Features

-   **Neural Network Model**: A simple feedforward neural network is built using PyTorch to approximate the velocity profile $u(r)$.
-   **Physics-Informed Loss Function**: The loss function is a composite of two parts:
    1.  **ODE Loss**: Enforces the governing differential equation at random collocation points within the pipe's radius.
    2.  **Boundary Condition (BC) Loss**: Enforces the no-slip condition ($u(R)=0$) at the pipe wall.
-   **Automatic Differentiation**: Utilizes PyTorch's `autograd` to compute the necessary derivatives ($\frac{du}{dr}$ and $\frac{d^2u}{dr^2}$) required for the ODE residual.
-   **Analytical Comparison**: The PINN's predicted solution is compared against the well-known analytical solution for Hagen-Poiseuille flow to validate its accuracy.
-   **Visualization**: The results are plotted to visually compare the PINN-predicted velocity profile with the exact analytical profile.

---

## 🚀 Getting Started

To run this project, you'll need Python with PyTorch and other common scientific computing libraries installed.

### Prerequisites

-   Python 3.8+
-   PyTorch
-   NumPy
-   Matplotlib

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install the required packages**:
    ```bash
    pip install torch numpy matplotlib
    ```

---

## 🛠️ Usage

1.  **Run the script from your terminal**:
    ```bash
    python main.py
    ```

2.  **Review the output**:
    -   The script will print the training loss at regular intervals.
    -   Upon completion, it will display a plot comparing the PINN's predicted velocity profile to the exact analytical solution.

---

## 💻 Code Structure

The project is contained within a single Python script (`main.py`) for ease of use. The main components are:

1.  **Problem Definition**:
    -   Constants for the pipe flow problem (radius `R`, pressure gradient `dP_dx`, viscosity `mu`) are defined.

2.  **PINN Model (`PINN` class)**:
    -   A `torch.nn.Module` subclass defining the neural network architecture (input layer, hidden layers with `tanh` activation, output layer).

3.  **Data Generation**:
    -   Collocation points (`r_collocation`) are randomly sampled within the pipe's domain (from 0 to R) to enforce the ODE.
    -   Boundary points (`r_boundary`) are defined at the pipe wall (r=R) to enforce the no-slip condition.

4.  **Training Loop**:
    -   The Adam optimizer is used to train the network.
    -   In each epoch, the model's prediction and its derivatives are computed.
    -   The composite loss (ODE residual + BC residual) is calculated and backpropagated to update the network's weights.

5.  **Evaluation and Plotting**:
    -   After training, the model predicts the velocity profile across a fine grid of radial points.
    -   The exact analytical solution is also calculated for comparison.
    -   Matplotlib is used to plot both profiles on the same graph for visual analysis.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss any changes or submit a pull request with your improvements.

---

## 📄 License

This project is licensed under the MIT License.
