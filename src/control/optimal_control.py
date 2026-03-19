"""
Optimal control methods for opinion dynamics.

Implements Forward-Backward Sweep (FBS) and Direct optimization methods
for computing optimal ranking gate policies.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Callable, Optional
from src.core.master_equation import MasterEquationSolver


class OptimalController:
    """
    Optimal control solver for opinion dynamics.
    """
    
    def __init__(
        self,
        solver: MasterEquationSolver,
        v: np.ndarray,
        K: float,
        Delta_min: float,
        Delta_max: float,
        Delta_star: float = 0.75
    ):
        """
        Initialize optimal controller.
        
        Parameters
        ----------
        solver : MasterEquationSolver
            Mean-field ODE solver
        v : np.ndarray
            Objective weights, shape (m,)
        K : float
            Running cost weight
        Delta_min : float
            Minimum ranking gate value
        Delta_max : float
            Maximum ranking gate value
        Delta_star : float
            Singular control default value
        """
        self.solver = solver
        self.v = v
        self.K = K
        self.Delta_min = Delta_min
        self.Delta_max = Delta_max
        self.Delta_star = Delta_star
        
        self.m = solver.m
        self.M = solver.M
    
    def compute_objective(
        self,
        tau_grid: np.ndarray,
        y_trajectory: np.ndarray,
        trapezoidal: bool = True
    ) -> float:
        """
        Compute objective function value.
        
        J = K * integral(sum_a v_a * y_a(tau) dtau) + sum_a v_a * y_a(T)
        
        Parameters
        ----------
        tau_grid : np.ndarray
            Time grid
        y_trajectory : np.ndarray
            State trajectory, shape (n_steps, m, M)
        trapezoidal : bool
            Use trapezoidal rule for integration
        
        Returns
        -------
        J : float
            Objective value
        """
        # Compute y_a(tau) = sum_f y_{a,f}(tau)
        y_agg = np.sum(y_trajectory, axis=2)  # Shape (n_steps, m)
        
        # Terminal cost
        J_terminal = np.dot(self.v, y_agg[-1])
        
        # Running cost
        if self.K > 0:
            if trapezoidal:
                # Trapezoidal rule
                dt = np.diff(tau_grid)
                integrand = np.dot(y_agg, self.v)
                J_running = 0.0
                for i in range(len(dt)):
                    J_running += 0.5 * dt[i] * (integrand[i] + integrand[i+1])
                J_running *= self.K
            else:
                # Rectangle rule
                dt = tau_grid[1] - tau_grid[0]
                integrand = np.dot(y_agg, self.v)
                J_running = self.K * dt * np.sum(integrand)
        else:
            J_running = 0.0
        
        return J_running + J_terminal
    
    def forward_backward_sweep(
        self,
        y0: np.ndarray,
        tau_grid: np.ndarray,
        Delta_init: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Forward-Backward Sweep algorithm for optimal control.
        
        Parameters
        ----------
        y0 : np.ndarray
            Initial state, shape (m, M)
        tau_grid : np.ndarray
            Time grid
        Delta_init : np.ndarray
            Initial control guess, shape (n_steps, m, m, M, M+1)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print iteration info
        
        Returns
        -------
        Delta_opt : np.ndarray
            Optimal control
        y_opt : np.ndarray
            Optimal trajectory
        J_opt : float
            Optimal objective value
        n_iter : int
            Number of iterations
        """
        n_steps = len(tau_grid)
        dt = tau_grid[1] - tau_grid[0] if len(tau_grid) > 1 else 1.0
        
        Delta = Delta_init.copy()
        J_prev = np.inf
        
        for iteration in range(max_iter):
            # Forward pass: integrate state equation
            y = np.zeros((n_steps, self.m, self.M))
            y[0] = y0.copy()
            
            for k in range(n_steps - 1):
                dydt = self.solver.compute_dynamics(y[k], Delta[k], tau_grid[k])
                y[k+1] = y[k] + dt * dydt
                
                # Project to feasible set
                y[k+1] = np.maximum(y[k+1], 0.0)
                for f in range(self.M):
                    total = np.sum(y[k+1, :, f])
                    if total > 1e-12:
                        y[k+1, :, f] *= self.solver.n[f] / total
            
            # Compute objective
            J = self.compute_objective(tau_grid, y, trapezoidal=True)
            
            if verbose:
                print(f"Iteration {iteration}: J = {J:.6f}")
            
            # Check convergence
            if iteration > 0 and abs(J - J_prev) / (abs(J_prev) + 1e-12) < tol:
                if verbose:
                    print(f"Converged in {iteration} iterations")
                return Delta, y, J, iteration
            
            J_prev = J
            
            # Backward pass: integrate adjoint equation
            lam = np.zeros((n_steps, self.m, self.M))
            lam[-1] = -self.v[:, np.newaxis].repeat(self.M, axis=1)
            
            for k in range(n_steps - 2, -1, -1):
                # Compute Jacobian at step k+1
                jac = self.compute_jacobian(y[k+1], Delta[k+1], tau_grid[k+1])
                
                # Backward Euler: lam[k] = lam[k+1] + dt * RHS
                if self.K > 0:
                    rhs = self.K * self.v[:, np.newaxis].repeat(self.M, axis=1) - jac.T @ lam[k+1].flatten()
                    rhs = rhs.reshape(self.m, self.M)
                else:
                    rhs = -jac.T @ lam[k+1].flatten()
                    rhs = rhs.reshape(self.m, self.M)
                
                lam[k] = lam[k+1] + dt * rhs
            
            # Update control: compute switching functions and apply bang-bang
            Delta_new = Delta.copy()
            # Determine control dimension based on Delta shape
            n_control_types = Delta.shape[-1]  # Could be M or M+1
            
            for k in range(n_steps):
                for s in range(self.m):
                    for l in range(self.m):
                        for f in range(self.M):
                            for r in range(n_control_types):
                                # Check bounds for P array
                                if r >= self.solver.P.shape[4]:
                                    continue
                                
                                # Switching function
                                Q = 0.0
                                for a in range(self.m):
                                    Q += lam[k, a, f] * self.solver.P[s, l, a, f, r]
                                Q -= lam[k, s, f]
                                
                                # Bang-bang control
                                if Q > 1e-8:
                                    Delta_new[k, s, l, f, r] = self.Delta_max
                                elif Q < -1e-8:
                                    Delta_new[k, s, l, f, r] = self.Delta_min
                                else:
                                    Delta_new[k, s, l, f, r] = self.Delta_star
            
            Delta = Delta_new
        
        if verbose:
            print(f"Maximum iterations {max_iter} reached")
        
        return Delta, y, J, max_iter
    
    def compute_jacobian(
        self,
        y: np.ndarray,
        Delta: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """
        Compute Jacobian of dynamics: dF/dy.
        
        Parameters
        ----------
        y : np.ndarray
            Current state, shape (m, M)
        Delta : np.ndarray
            Current control
        tau : float
            Current time
        
        Returns
        -------
        jac : np.ndarray
            Jacobian matrix, shape (m*M, m*M)
        """
        jac = np.zeros((self.m * self.M, self.m * self.M))
        B = self.solver.compute_B(tau)
        
        for a in range(self.m):
            for f in range(self.M):
                if B[f] < 1e-12:
                    continue
                
                idx_af = a * self.M + f
                
                for i in range(self.m):
                    for j in range(self.M):
                        idx_ij = i * self.M + j
                        
                        deriv = 0.0
                        
                        # Derivative w.r.t. y_{i,j}
                        if f == j:
                            # y_{i,j} appears as y_{s,f} in the gain term
                            for l in range(self.m):
                                for r in range(self.M):
                                    weight = y[l, r] * self.solver.Omega[f, r] * Delta[i, l, f, r]
                                    deriv += weight * self.solver.P[i, l, a, f, r]
                                    if i == a:
                                        deriv -= weight
                                
                                # Bot contribution
                                if self.solver.u > 0:
                                    u_lf = self.solver.u_bot[l, f]
                                    if u_lf > 1e-12:
                                        weight = u_lf * self.solver.Omega_bot[f] * Delta[i, l, f, self.M]
                                        deriv += weight * self.solver.P[i, l, a, f, self.M]
                                        if i == a:
                                            deriv -= weight
                        
                        # y_{i,j} appears as y_{l,r} in source terms
                        for s in range(self.m):
                            weight = y[s, f] * self.solver.Omega[f, j] * Delta[s, i, f, j]
                            deriv += weight * self.solver.P[s, i, a, f, j]
                            if s == a:
                                deriv -= weight
                        
                        jac[idx_af, idx_ij] = (self.solver.pi[f] / (self.solver.A * B[f])) * deriv
        
        return jac
    
    def direct_optimization(
        self,
        y0: np.ndarray,
        tau_grid: np.ndarray,
        Delta_init: np.ndarray,
        method: str = "L-BFGS-B",
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Direct optimization method for optimal control.
        
        Parameters
        ----------
        y0 : np.ndarray
            Initial state
        tau_grid : np.ndarray
            Time grid
        Delta_init : np.ndarray
            Initial control guess
        method : str
            Optimization method for scipy.optimize.minimize
        verbose : bool
            Print optimization info
        
        Returns
        -------
        Delta_opt : np.ndarray
            Optimal control
        y_opt : np.ndarray
            Optimal trajectory
        J_opt : float
            Optimal objective value
        """
        n_steps = len(tau_grid)
        dt = tau_grid[1] - tau_grid[0] if len(tau_grid) > 1 else 1.0
        
        # Flatten control for optimization
        Delta_shape = Delta_init.shape
        n_vars = np.prod(Delta_shape)
        
        def objective(Delta_flat):
            Delta = Delta_flat.reshape(Delta_shape)
            
            # Forward integrate
            y = np.zeros((n_steps, self.m, self.M))
            y[0] = y0.copy()
            
            for k in range(n_steps - 1):
                dydt = self.solver.compute_dynamics(y[k], Delta[k], tau_grid[k])
                y[k+1] = y[k] + dt * dydt
                
                # Project to feasible set
                y[k+1] = np.maximum(y[k+1], 0.0)
                for f in range(self.M):
                    total = np.sum(y[k+1, :, f])
                    if total > 1e-12:
                        y[k+1, :, f] *= self.solver.n[f] / total
            
            # Compute objective
            J = self.compute_objective(tau_grid, y, trapezoidal=True)
            return J
        
        # Bounds
        bounds = [(self.Delta_min, self.Delta_max)] * n_vars
        
        # Optimize
        result = minimize(
            objective,
            Delta_init.flatten(),
            method=method,
            bounds=bounds,
            options={'disp': verbose, 'maxiter': 1000}
        )
        
        Delta_opt = result.x.reshape(Delta_shape)
        
        # Compute optimal trajectory
        y_opt = np.zeros((n_steps, self.m, self.M))
        y_opt[0] = y0.copy()
        
        for k in range(n_steps - 1):
            dydt = self.solver.compute_dynamics(y_opt[k], Delta_opt[k], tau_grid[k])
            y_opt[k+1] = y_opt[k] + dt * dydt
            y_opt[k+1] = np.maximum(y_opt[k+1], 0.0)
            for f in range(self.M):
                total = np.sum(y_opt[k+1, :, f])
                if total > 1e-12:
                    y_opt[k+1, :, f] *= self.solver.n[f] / total
        
        J_opt = result.fun
        
        return Delta_opt, y_opt, J_opt


def create_analytical_controller_m2(
    v: np.ndarray,
    Delta_min: float,
    Delta_max: float,
    m: int,
    M: int
) -> np.ndarray:
    """
    Create analytical optimal controller for m=2 (Theorem 1, Eq. 26).
    
    For minimizing opinion Z_2 (v = [0, 1]):
    - Delta_{s,1}^{f,r} = Delta_min for all s (protect Z_1 sources)
    - Delta_{s,2}^{f,r} = Delta_max for all s (expose Z_2 sources)
    
    Parameters
    ----------
    v : np.ndarray
        Objective weights
    Delta_min : float
        Minimum gate value
    Delta_max : float
        Maximum gate value
    m : int
        Number of opinions (must be 2)
    M : int
        Number of native types
    
    Returns
    -------
    Delta : np.ndarray
        Analytical controller, shape (m, m, M, M+1)
    """
    assert m == 2, "Analytical controller only for m=2"
    
    Delta = np.zeros((m, m, M, M + 1))
    
    # Determine which opinion to minimize (higher weight)
    target_opinion = np.argmax(v)
    
    for s in range(m):
        for f in range(M):
            for r in range(M + 1):
                # Minimize exposure to target opinion sources
                Delta[s, target_opinion, f, r] = Delta_max
                Delta[s, 1 - target_opinion, f, r] = Delta_min
    
    return Delta
