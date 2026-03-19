"""
Master equation (mean-field ODE) solver for opinion dynamics.

This module implements the nonlinear mean-field approximation of the agent-based model
following the master equation formulation from the paper.
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple


class MasterEquationSolver:
    """
    Solver for the master equation (mean-field ODE) of opinion dynamics.
    
    The master equation tracks the macroscopic opinion fractions y_{a,f}(tau) 
    where a is opinion index and f is native type index.
    """
    
    def __init__(
        self,
        m: int,
        M: int,
        n: np.ndarray,
        u: float,
        pi: np.ndarray,
        rho: np.ndarray,
        P: np.ndarray,
        u_bot: Optional[np.ndarray] = None,
        rho_bot: Optional[np.ndarray] = None,
        Omega_bot: Optional[np.ndarray] = None
    ):
        """
        Initialize master equation solver.
        
        Parameters
        ----------
        m : int
            Number of opinions
        M : int
            Number of native types
        n : np.ndarray
            Native type fractions, shape (M,)
        u : float
            Bot fraction
        pi : np.ndarray
            Activity rates, shape (M+1,) where pi[M] is bot activity rate
        rho : np.ndarray
            SBM link probabilities between native types, shape (M, M)
        P : np.ndarray
            Transition probability tables, shape (m, m, m, M+1, M+1)
            P[s, l, k, f, r] = probability to adopt opinion k given object opinion s,
            source opinion l, object type f, source type r
        u_bot : np.ndarray, optional
            Bot strategy matrix, shape (m, M), u_bot[l, f] = fraction of bots 
            targeting type f with opinion l
        rho_bot : np.ndarray, optional
            Native-bot link intensities, shape (M,), rho_bot[f] for type f
        Omega_bot : np.ndarray, optional
            Precomputed Omega_f for native-bot interactions, shape (M,)
        """
        self.m = m
        self.M = M
        self.n = n
        self.u = u
        self.pi = pi
        self.rho = rho
        self.P = P
        
        # Bot parameters (can be None if no bots)
        self.u_bot = u_bot if u_bot is not None else np.zeros((m, M))
        self.rho_bot = rho_bot if rho_bot is not None else np.zeros(M)
        self.Omega_bot = Omega_bot if Omega_bot is not None else np.zeros(M)
        
        # Compute normalization constants
        self.A = np.sum(n * pi[:M]) + u * pi[M]
        
        # Compute Omega_{f,r} = pi_r * rho_{f,r}
        self.Omega = np.zeros((M, M))
        for f in range(M):
            for r in range(M):
                self.Omega[f, r] = pi[r] * rho[f, r]
        
        # Compute B_f for each type f
        self.compute_B()
    
    def compute_B(self, tau: float = 0.0) -> np.ndarray:
        """
        Compute B_f(tau) = sum_r n_r * Omega_{f,r} + u_f(tau) * Omega_f(tau).
        
        Parameters
        ----------
        tau : float
            Current time (for time-dependent bot strategies)
        
        Returns
        -------
        B : np.ndarray
            Array of B_f values, shape (M,)
        """
        B = np.zeros(self.M)
        for f in range(self.M):
            # Native-native contribution
            B[f] = np.sum(self.n * self.Omega[f, :])
            # Native-bot contribution
            u_f = np.sum(self.u_bot[:, f])
            B[f] += u_f * self.Omega_bot[f]
        return B
    
    def compute_dynamics(
        self,
        y: np.ndarray,
        Delta: np.ndarray,
        tau: float = 0.0
    ) -> np.ndarray:
        """
        Compute dy/dtau according to the master equation.
        
        Parameters
        ----------
        y : np.ndarray
            Current state, shape (m, M)
        Delta : np.ndarray
            Ranking gate parameters, shape (m, m, M, M+1)
            Delta[s, l, f, r] = gate for object opinion s, source opinion l,
            object type f, source type r (r < M for natives, r = M for bots)
        tau : float
            Current time
        
        Returns
        -------
        dydt : np.ndarray
            Time derivative, shape (m, M)
        """
        dydt = np.zeros((self.m, self.M))
        B = self.compute_B(tau)
        
        for a in range(self.m):
            for f in range(self.M):
                if B[f] < 1e-12:
                    continue
                
                gain = 0.0
                loss = 0.0
                
                # Iterate over object opinions and source opinions
                for s in range(self.m):
                    for l in range(self.m):
                        # Native-native interactions
                        for r in range(self.M):
                            weight = y[s, f] * y[l, r] * self.Omega[f, r] * Delta[s, l, f, r]
                            gain += weight * self.P[s, l, a, f, r]
                            if s == a:
                                loss += weight
                        
                        # Native-bot interactions
                        if self.u > 0:
                            u_lf = self.u_bot[l, f]
                            if u_lf > 1e-12:
                                weight = y[s, f] * u_lf * self.Omega_bot[f] * Delta[s, l, f, self.M]
                                gain += weight * self.P[s, l, a, f, self.M]
                                if s == a:
                                    loss += weight
                
                dydt[a, f] = (self.pi[f] / (self.A * B[f])) * (gain - loss)
        
        return dydt
    
    def integrate(
        self,
        y0: np.ndarray,
        Delta: Callable[[float], np.ndarray],
        T: float,
        dt: float = 0.01,
        record_every: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate master equation using forward Euler method.
        
        Parameters
        ----------
        y0 : np.ndarray
            Initial state, shape (m, M)
        Delta : callable
            Function tau -> Delta(tau), returns ranking gate array
        T : float
            Final time
        dt : float
            Time step size
        record_every : int
            Record state every n steps
        
        Returns
        -------
        tau_array : np.ndarray
            Time points
        y_array : np.ndarray
            State trajectory, shape (n_steps, m, M)
        """
        n_steps = int(T / dt)
        n_records = n_steps // record_every + 1
        
        tau_array = np.zeros(n_records)
        y_array = np.zeros((n_records, self.m, self.M))
        
        y = y0.copy()
        tau_array[0] = 0.0
        y_array[0] = y.copy()
        
        record_idx = 1
        for step in range(n_steps):
            tau = step * dt
            Delta_current = Delta(tau)
            dydt = self.compute_dynamics(y, Delta_current, tau)
            y = y + dt * dydt
            
            # Ensure non-negativity and mass conservation
            y = np.maximum(y, 0.0)
            for f in range(self.M):
                total = np.sum(y[:, f])
                if total > 1e-12:
                    y[:, f] *= self.n[f] / total
            
            if (step + 1) % record_every == 0 and record_idx < n_records:
                tau_array[record_idx] = tau + dt
                y_array[record_idx] = y.copy()
                record_idx += 1
        
        return tau_array, y_array


def create_constant_Delta(
    m: int,
    M: int,
    value: float,
    include_bots: bool = True
) -> np.ndarray:
    """
    Create a constant ranking gate array.
    
    Parameters
    ----------
    m : int
        Number of opinions
    M : int
        Number of native types
    value : float
        Constant value for all gates
    include_bots : bool
        If True, include bot type (dimension M+1), else dimension M
    
    Returns
    -------
    Delta : np.ndarray
        Constant gate array
    """
    n_types = M + 1 if include_bots else M
    Delta = np.full((m, m, M, n_types), value, dtype=np.float64)
    return Delta


def create_type_homophily_Delta(
    m: int,
    M: int,
    Delta_high: float,
    Delta_low: float,
    include_bots: bool = True
) -> np.ndarray:
    """
    Create ranking gate with type homophily: Delta_{s,l}^{f,r} = Delta_high if f==r else Delta_low.
    
    Parameters
    ----------
    m : int
        Number of opinions
    M : int
        Number of native types
    Delta_high : float
        Gate value for same type
    Delta_low : float
        Gate value for different type
    include_bots : bool
        If True, include bot type
    
    Returns
    -------
    Delta : np.ndarray
        Type homophily gate array
    """
    n_types = M + 1 if include_bots else M
    Delta = np.full((m, m, M, n_types), Delta_low, dtype=np.float64)
    
    for f in range(M):
        Delta[:, :, f, f] = Delta_high
    
    return Delta


def create_opinion_heterophily_Delta(
    m: int,
    M: int,
    Delta_high: float,
    Delta_low: float,
    include_bots: bool = True
) -> np.ndarray:
    """
    Create ranking gate with opinion heterophily: Delta_{s,l}^{f,r} = Delta_high if s!=l else Delta_low.
    
    Parameters
    ----------
    m : int
        Number of opinions
    M : int
        Number of native types
    Delta_high : float
        Gate value for different opinion
    Delta_low : float
        Gate value for same opinion
    include_bots : bool
        If True, include bot type
    
    Returns
    -------
    Delta : np.ndarray
        Opinion heterophily gate array
    """
    n_types = M + 1 if include_bots else M
    Delta = np.full((m, m, M, n_types), 0.0, dtype=np.float64)
    
    for s in range(m):
        for l in range(m):
            if s != l:
                Delta[s, l, :, :] = Delta_high
            else:
                Delta[s, l, :, :] = Delta_low
    
    return Delta
