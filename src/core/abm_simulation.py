"""
Agent-based model (ABM) simulation for opinion dynamics.

This module implements stochastic simulations of individual agents on a network
following the microscopic dynamics described in the paper.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from numba import jit


class ABMSimulator:
    """
    Agent-based model simulator for opinion dynamics with ranking gates.
    """
    
    def __init__(
        self,
        m: int,
        M: int,
        N: int,
        N_types: np.ndarray,
        U: int,
        pi: np.ndarray,
        rho: np.ndarray,
        P: np.ndarray,
        u_bot: Optional[np.ndarray] = None,
        rho_bot: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        """
        Initialize ABM simulator.
        
        Parameters
        ----------
        m : int
            Number of opinions
        M : int
            Number of native types
        N : int
            Total number of native agents
        N_types : np.ndarray
            Number of agents per type, shape (M,)
        U : int
            Number of bot agents
        pi : np.ndarray
            Activity rates, shape (M+1,) where pi[M] is bot activity rate
        rho : np.ndarray
            SBM link probabilities, shape (M, M)
        P : np.ndarray
            Transition probabilities, shape (m, m, m, M+1, M+1)
        u_bot : np.ndarray, optional
            Bot strategy matrix, shape (m, M)
        rho_bot : np.ndarray, optional
            Native-bot link intensities, shape (M,)
        seed : int
            Random seed
        """
        self.m = m
        self.M = M
        self.N = N
        self.N_types = N_types
        self.U = U
        self.pi = pi
        self.rho = rho
        self.P = P
        self.u_bot = u_bot if u_bot is not None else np.zeros((m, M))
        self.rho_bot = rho_bot if rho_bot is not None else np.zeros(M)
        self.seed = seed
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
        # Agent arrays
        self.x = np.zeros(N + U, dtype=np.int32)  # Opinions
        self.xi = np.zeros(N + U, dtype=np.int32)  # Types
        
        # Initialize types
        idx = 0
        for f in range(M):
            for i in range(N_types[f]):
                self.xi[idx] = f
                idx += 1
        
        # Bots have type M
        for i in range(U):
            self.xi[N + i] = M
        
        # Build adjacency lists for efficiency
        self.adjacency = None
    
    def initialize_opinions(self, q: np.ndarray) -> None:
        """
        Initialize agent opinions according to initial distribution.
        
        Parameters
        ----------
        q : np.ndarray
            Initial joint state matrix, shape (m, M)
            q[a, f] = fraction of type f agents with opinion a
        """
        idx = 0
        for f in range(self.M):
            N_f = self.N_types[f]
            # Sample opinions for type f agents
            probs = q[:, f] / np.sum(q[:, f]) if np.sum(q[:, f]) > 0 else np.ones(self.m) / self.m
            opinions = self.rng.choice(self.m, size=N_f, p=probs)
            for i in range(N_f):
                self.x[idx] = opinions[i]
                idx += 1
        
        # Initialize bot opinions (fixed to a specific opinion, typically Z_3 = index 2)
        for i in range(self.U):
            self.x[self.N + i] = 2  # Bots hold opinion Z_3
    
    def build_network(self) -> None:
        """
        Build static SBM network for native-native and native-bot edges.
        Uses adjacency lists for efficiency.
        """
        self.adjacency = [[] for _ in range(self.N + self.U)]
        
        # Native-native edges
        for i in range(self.N):
            for j in range(i + 1, self.N):
                f = self.xi[i]
                r = self.xi[j]
                p_edge = self.rho[f, r] / self.N
                if self.rng.rand() < p_edge:
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)
        
        # Native-bot edges
        for i in range(self.N):
            f = self.xi[i]
            for b in range(self.N, self.N + self.U):
                p_edge = self.rho_bot[f] / self.N
                if self.rng.rand() < p_edge:
                    self.adjacency[i].append(b)
                    self.adjacency[b].append(i)
    
    def simulate(
        self,
        q0: np.ndarray,
        Delta: Callable[[float], np.ndarray],
        T_tau: float,
        record_every_tau: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ABM simulation.
        
        Parameters
        ----------
        q0 : np.ndarray
            Initial joint state matrix, shape (m, M)
        Delta : callable
            Function tau -> Delta(tau), returns ranking gate array
        T_tau : float
            Final time in tau units (tau = t / N)
        record_every_tau : float
            Record state every n tau units
        
        Returns
        -------
        tau_array : np.ndarray
            Time points in tau units
        y_array : np.ndarray
            State trajectory, shape (n_records, m, M)
        """
        # Initialize
        self.initialize_opinions(q0)
        self.build_network()
        
        # Compute total activity for selection probabilities
        total_activity = np.sum(self.pi[:self.M] * self.N_types) + self.pi[self.M] * self.U
        selection_probs = np.zeros(self.N + self.U)
        for i in range(self.N + self.U):
            selection_probs[i] = self.pi[self.xi[i]]
        selection_probs /= np.sum(selection_probs)
        
        # Simulation parameters
        T_steps = int(T_tau * self.N)
        record_every = int(record_every_tau * self.N)
        n_records = T_steps // record_every + 1
        
        tau_array = np.zeros(n_records)
        y_array = np.zeros((n_records, self.m, self.M))
        
        # Record initial state
        tau_array[0] = 0.0
        y_array[0] = self.compute_macrostate()
        
        record_idx = 1
        
        # Main simulation loop
        for t in range(1, T_steps + 1):
            tau = t / self.N
            Delta_current = Delta(tau)
            
            # Sample influence object
            i = self.rng.choice(self.N + self.U, p=selection_probs)
            
            # Bots are invulnerable
            if i >= self.N:
                continue
            
            # Check if agent has neighbors
            if len(self.adjacency[i]) == 0:
                continue
            
            # Sample influence source from neighbors
            neighbors = self.adjacency[i]
            neighbor_activities = np.array([self.pi[self.xi[j]] for j in neighbors])
            neighbor_probs = neighbor_activities / np.sum(neighbor_activities)
            j = self.rng.choice(neighbors, p=neighbor_probs)
            
            # Ranking gate
            s = self.x[i]
            l = self.x[j]
            f = self.xi[i]
            r = self.xi[j]
            
            gate = Delta_current[s, l, f, r]
            
            # Accept interaction with probability gate
            if self.rng.rand() < gate:
                # Update opinion
                probs = self.P[s, l, :, f, r]
                k = self.rng.choice(self.m, p=probs)
                self.x[i] = k
            
            # Record state
            if t % record_every == 0 and record_idx < n_records:
                tau_array[record_idx] = tau
                y_array[record_idx] = self.compute_macrostate()
                record_idx += 1
        
        return tau_array[:record_idx], y_array[:record_idx]
    
    def compute_macrostate(self) -> np.ndarray:
        """
        Compute macroscopic state y_{a,f} = fraction of type f with opinion a.
        
        Returns
        -------
        y : np.ndarray
            Macrostate, shape (m, M)
        """
        y = np.zeros((self.m, self.M))
        
        for i in range(self.N):
            a = self.x[i]
            f = self.xi[i]
            y[a, f] += 1.0
        
        # Normalize by type counts
        for f in range(self.M):
            if self.N_types[f] > 0:
                y[:, f] /= self.N_types[f]
        
        return y


def run_monte_carlo_simulations(
    n_runs: int,
    simulator_factory: Callable[[int], ABMSimulator],
    q0: np.ndarray,
    Delta: Callable[[float], np.ndarray],
    T_tau: float,
    record_every_tau: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run multiple Monte Carlo simulations and compute statistics.
    
    Parameters
    ----------
    n_runs : int
        Number of independent runs
    simulator_factory : callable
        Function (seed) -> ABMSimulator that creates a new simulator
    q0 : np.ndarray
        Initial state
    Delta : callable
        Ranking gate function
    T_tau : float
        Final time in tau units
    record_every_tau : float
        Recording interval
    
    Returns
    -------
    tau_array : np.ndarray
        Time points
    y_mean : np.ndarray
        Mean trajectory across runs
    y_std : np.ndarray
        Standard deviation across runs
    y_all : np.ndarray
        All trajectories, shape (n_runs, n_records, m, M)
    """
    # Run first simulation to get dimensions
    sim = simulator_factory(0)
    tau_array, y0 = sim.simulate(q0, Delta, T_tau, record_every_tau)
    
    n_records, m, M = y0.shape
    y_all = np.zeros((n_runs, n_records, m, M))
    y_all[0] = y0
    
    # Run remaining simulations
    for run in range(1, n_runs):
        sim = simulator_factory(run)
        _, y = sim.simulate(q0, Delta, T_tau, record_every_tau)
        # Handle different lengths due to simulation termination timing
        if y.shape[0] != n_records:
            # Pad or truncate to match
            if y.shape[0] < n_records:
                y_padded = np.zeros((n_records, m, M))
                y_padded[:y.shape[0]] = y
                # Fill remaining with last value
                for i in range(y.shape[0], n_records):
                    y_padded[i] = y[-1]
                y = y_padded
            else:
                y = y[:n_records]
        y_all[run] = y
    
    # Compute statistics
    y_mean = np.mean(y_all, axis=0)
    y_std = np.std(y_all, axis=0)
    
    return tau_array, y_mean, y_std, y_all
