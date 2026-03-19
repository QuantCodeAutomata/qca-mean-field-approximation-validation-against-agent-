"""Tests for master equation solver."""

import numpy as np
import pytest
from src.core.master_equation import (
    MasterEquationSolver,
    create_constant_Delta,
    create_type_homophily_Delta,
    create_opinion_heterophily_Delta
)
from src.core.transition_tables import create_transition_tensor


def test_master_equation_initialization():
    """Test that MasterEquationSolver initializes correctly."""
    m, M = 2, 2
    n = np.array([0.5, 0.5])
    u = 0.0
    pi = np.array([1.0, 1.0, 1.0])
    rho = np.array([[0.5, 0.2], [0.2, 0.5]])
    P = create_transition_tensor(m, M, include_bots=False)
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P)
    
    assert solver.m == m
    assert solver.M == M
    assert np.allclose(solver.n, n)
    assert solver.u == u


def test_mass_conservation():
    """Test that mass is conserved during integration."""
    m, M = 2, 2
    n = np.array([0.6, 0.4])
    u = 0.0
    pi = np.array([1.0, 1.0, 1.0])
    rho = np.array([[0.5, 0.2], [0.2, 0.5]])
    P = create_transition_tensor(m, M, include_bots=False)
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P)
    
    # Initial state
    y0 = np.array([[0.3, 0.2], [0.3, 0.2]])
    
    # Create constant Delta
    Delta = create_constant_Delta(m, M, 1.0, include_bots=False)
    Delta_func = lambda tau: Delta
    
    # Integrate
    tau_array, y_array = solver.integrate(y0, Delta_func, T=10.0, dt=0.01)
    
    # Check mass conservation at each time step
    for i in range(len(tau_array)):
        for f in range(M):
            mass = np.sum(y_array[i, :, f])
            assert np.abs(mass - n[f]) < 1e-6, f"Mass not conserved at step {i}, type {f}: {mass} vs {n[f]}"


def test_non_negativity():
    """Test that state variables remain non-negative."""
    m, M = 3, 2
    n = np.array([0.5, 0.5])
    u = 0.0
    pi = np.array([1.0, 1.0, 1.0])
    rho = np.array([[0.4, 0.1], [0.1, 0.4]])
    P = create_transition_tensor(m, M, include_bots=False, alpha=0.6)
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P)
    
    # Initial state
    y0 = np.array([[0.2, 0.2], [0.2, 0.2], [0.1, 0.1]])
    
    Delta = create_constant_Delta(m, M, 1.0, include_bots=False)
    Delta_func = lambda tau: Delta
    
    tau_array, y_array = solver.integrate(y0, Delta_func, T=20.0, dt=0.01)
    
    # Check non-negativity
    assert np.all(y_array >= -1e-10), "State variables became negative"


def test_constant_Delta_creation():
    """Test creation of constant Delta arrays."""
    m, M = 3, 2
    value = 0.75
    
    Delta = create_constant_Delta(m, M, value, include_bots=True)
    
    assert Delta.shape == (m, m, M, M+1)
    assert np.allclose(Delta, value)


def test_type_homophily_Delta():
    """Test type homophily Delta structure."""
    m, M = 2, 2
    Delta_high, Delta_low = 1.0, 0.5
    
    Delta = create_type_homophily_Delta(m, M, Delta_high, Delta_low, include_bots=False)
    
    # Check within-type interactions are high
    for f in range(M):
        assert np.allclose(Delta[:, :, f, f], Delta_high)
    
    # Check cross-type interactions are low
    for f in range(M):
        for r in range(M):
            if f != r:
                assert np.allclose(Delta[:, :, f, r], Delta_low)


def test_opinion_heterophily_Delta():
    """Test opinion heterophily Delta structure."""
    m, M = 3, 2
    Delta_high, Delta_low = 1.0, 0.3
    
    Delta = create_opinion_heterophily_Delta(m, M, Delta_high, Delta_low, include_bots=False)
    
    # Check same opinion pairs have low Delta
    for s in range(m):
        assert np.allclose(Delta[s, s, :, :], Delta_low)
    
    # Check different opinion pairs have high Delta
    for s in range(m):
        for l in range(m):
            if s != l:
                assert np.allclose(Delta[s, l, :, :], Delta_high)


def test_equilibrium_convergence():
    """Test that homogeneous initial state remains at equilibrium."""
    m, M = 2, 1
    n = np.array([1.0])
    u = 0.0
    pi = np.array([1.0, 1.0])
    rho = np.array([[1.0]])
    P = create_transition_tensor(m, M, include_bots=False)
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P)
    
    # Uniform initial state
    y0 = np.array([[0.5], [0.5]])
    
    Delta = create_constant_Delta(m, M, 1.0, include_bots=False)
    Delta_func = lambda tau: Delta
    
    tau_array, y_array = solver.integrate(y0, Delta_func, T=10.0, dt=0.01)
    
    # State should remain approximately constant (symmetric dynamics)
    assert np.allclose(y_array[0], y_array[-1], atol=0.1)


def test_B_computation():
    """Test computation of B_f values."""
    m, M = 2, 2
    n = np.array([0.6, 0.4])
    u = 0.1
    pi = np.array([2.0, 1.0, 3.0])
    rho = np.array([[0.4, 0.1], [0.1, 0.4]])
    P = create_transition_tensor(m, M, include_bots=True)
    
    u_bot = np.zeros((m, M))
    u_bot[1, 0] = 0.05  # Some bots target type 0
    u_bot[1, 1] = 0.05  # Some bots target type 1
    
    rho_bot = np.array([0.2, 0.2])
    Omega_bot = pi[2] * rho_bot
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P, u_bot, rho_bot, Omega_bot)
    
    B = solver.compute_B(tau=0.0)
    
    # B should be positive for all types
    assert np.all(B > 0), "B values should be positive"
    
    # Manual computation check for type 0
    B0_expected = n[0] * pi[0] * rho[0, 0] + n[1] * pi[1] * rho[0, 1] + np.sum(u_bot[:, 0]) * Omega_bot[0]
    assert np.abs(B[0] - B0_expected) < 1e-10, f"B[0] mismatch: {B[0]} vs {B0_expected}"


def test_dynamics_computation():
    """Test that dynamics computation runs without errors."""
    m, M = 3, 2
    n = np.array([0.5, 0.5])
    u = 0.0
    pi = np.array([1.0, 1.0, 1.0])
    rho = np.array([[0.4, 0.1], [0.1, 0.4]])
    P = create_transition_tensor(m, M, include_bots=False)
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P)
    
    y = np.array([[0.2, 0.2], [0.2, 0.2], [0.1, 0.1]])
    Delta = create_constant_Delta(m, M, 1.0, include_bots=False)
    
    dydt = solver.compute_dynamics(y, Delta, tau=0.0)
    
    assert dydt.shape == (m, M)
    assert np.all(np.isfinite(dydt)), "Dynamics contain non-finite values"


def test_integration_step_size():
    """Test that different step sizes produce similar results."""
    m, M = 2, 1
    n = np.array([1.0])
    u = 0.0
    pi = np.array([1.0, 1.0])
    rho = np.array([[0.5]])
    P = create_transition_tensor(m, M, include_bots=False)
    
    solver = MasterEquationSolver(m, M, n, u, pi, rho, P)
    
    y0 = np.array([[0.7], [0.3]])
    Delta = create_constant_Delta(m, M, 1.0, include_bots=False)
    Delta_func = lambda tau: Delta
    
    # Coarse integration
    tau1, y1 = solver.integrate(y0, Delta_func, T=5.0, dt=0.1, record_every=10)
    
    # Fine integration
    tau2, y2 = solver.integrate(y0, Delta_func, T=5.0, dt=0.01, record_every=100)
    
    # Interpolate to compare
    y2_interp = np.zeros_like(y1)
    for a in range(m):
        for f in range(M):
            y2_interp[:, a, f] = np.interp(tau1, tau2, y2[:, a, f])
    
    # Should be reasonably close
    assert np.allclose(y1, y2_interp, atol=0.05), "Results differ significantly with step size"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
