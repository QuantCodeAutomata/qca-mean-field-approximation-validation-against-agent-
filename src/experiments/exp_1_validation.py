"""
Experiment 1: Mean-Field Approximation Validation

Validate that the nonlinear mean-field ODE accurately tracks the macroscopic 
opinion fractions of the underlying stochastic agent-based model under three 
ranking regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict

from src.core.master_equation import (
    MasterEquationSolver,
    create_constant_Delta,
    create_type_homophily_Delta,
    create_opinion_heterophily_Delta
)
from src.core.abm_simulation import ABMSimulator, run_monte_carlo_simulations
from src.core.transition_tables import create_transition_tensor


def setup_parameters() -> Dict:
    """Set up parameters for Experiment 1 as specified in the paper."""
    # Note: Using reduced scale for demonstration (paper uses N=5000, T=75)
    params = {
        'm': 3,  # Number of opinions
        'M': 2,  # Number of native types
        'N': 500,  # Total native agents (reduced from 5000 for speed)
        'N1': 250,  # Type 1 agents
        'N2': 200,  # Type 2 agents
        'U': 50,  # Bot agents
        'u': 0.1,  # Bot fraction
        'n1': 0.5,  # Type 1 fraction
        'n2': 0.4,  # Type 2 fraction
        'T_tau': 30.0,  # Final time in tau units (reduced from 75)
        'record_every_tau': 1.0,  # Recording interval
        'n_monte_carlo': 5,  # Number of Monte Carlo runs (reduced from 10)
    }
    
    # Initial joint state matrix q (m x M)
    params['q'] = np.array([
        [0.4, 0.0],  # Opinion Z_1
        [0.1, 0.1],  # Opinion Z_2
        [0.0, 0.3],  # Opinion Z_3
    ])
    
    # SBM link probabilities
    params['rho'] = np.array([
        [0.4, 0.1],
        [0.1, 0.4],
    ])
    
    # Activity rates (native type 1, native type 2, bots)
    params['pi'] = np.array([2.0, 1.0, 3.0])
    
    # Bot strategy: u_{3,1} = 0.1 (bots target type 1, hold opinion Z_3)
    params['u_bot'] = np.zeros((3, 2))
    params['u_bot'][2, 0] = 0.1  # Opinion 3, type 1
    
    # Native-bot link intensity
    params['rho_bot'] = np.array([0.1, 0.1])  # For both types
    
    # Compute Omega_bot
    params['Omega_bot'] = params['pi'][2] * params['rho_bot']
    
    # Type fractions
    params['n'] = np.array([params['n1'], params['n2']])
    
    # Type counts
    params['N_types'] = np.array([params['N1'], params['N2']])
    
    # Ranking parameters
    params['Delta_high'] = 1.0
    params['Delta_low'] = 0.5
    
    return params


def run_experiment_1() -> Dict:
    """
    Run Experiment 1: MFA validation against ABM for three ranking regimes.
    
    Returns
    -------
    results : dict
        Dictionary containing all results
    """
    print("=" * 80)
    print("EXPERIMENT 1: Mean-Field Approximation Validation")
    print("=" * 80)
    
    params = setup_parameters()
    
    # Create transition probability tensor
    P = create_transition_tensor(
        m=params['m'],
        M=params['M'],
        include_bots=True,
        table_type="symmetric",
        alpha=0.5,
        beta=0.3
    )
    
    results = {}
    regimes = ['no_ranking', 'type_homophily', 'opinion_heterophily']
    
    for regime in regimes:
        print(f"\n{'-' * 80}")
        print(f"Running regime: {regime}")
        print(f"{'-' * 80}")
        
        # Create ranking gate
        if regime == 'no_ranking':
            Delta = create_constant_Delta(params['m'], params['M'], 1.0, include_bots=True)
        elif regime == 'type_homophily':
            Delta = create_type_homophily_Delta(
                params['m'], params['M'],
                params['Delta_high'], params['Delta_low'],
                include_bots=True
            )
        else:  # opinion_heterophily
            Delta = create_opinion_heterophily_Delta(
                params['m'], params['M'],
                params['Delta_high'], params['Delta_low'],
                include_bots=True
            )
        
        Delta_func = lambda tau: Delta
        
        # Run Mean-Field Approximation
        print("Running MFA...")
        solver = MasterEquationSolver(
            m=params['m'],
            M=params['M'],
            n=params['n'],
            u=params['u'],
            pi=params['pi'],
            rho=params['rho'],
            P=P,
            u_bot=params['u_bot'],
            rho_bot=params['rho_bot'],
            Omega_bot=params['Omega_bot']
        )
        
        tau_mfa, y_mfa = solver.integrate(
            y0=params['q'],
            Delta=Delta_func,
            T=params['T_tau'],
            dt=0.01,
            record_every=100  # Record every tau unit
        )
        
        # Aggregate over types
        y_mfa_agg = np.sum(y_mfa, axis=2)  # Shape (n_steps, m)
        
        # Run Agent-Based Model (Monte Carlo)
        print(f"Running ABM with {params['n_monte_carlo']} Monte Carlo runs...")
        
        def simulator_factory(seed):
            return ABMSimulator(
                m=params['m'],
                M=params['M'],
                N=params['N'],
                N_types=params['N_types'],
                U=params['U'],
                pi=params['pi'],
                rho=params['rho'],
                P=P,
                u_bot=params['u_bot'],
                rho_bot=params['rho_bot'],
                seed=seed
            )
        
        tau_abm, y_abm_mean, y_abm_std, y_abm_all = run_monte_carlo_simulations(
            n_runs=params['n_monte_carlo'],
            simulator_factory=simulator_factory,
            q0=params['q'],
            Delta=Delta_func,
            T_tau=params['T_tau'],
            record_every_tau=params['record_every_tau']
        )
        
        # Aggregate ABM over types
        y_abm_mean_agg = np.sum(y_abm_mean, axis=2)  # Shape (n_steps, m)
        y_abm_std_agg = np.sqrt(np.sum(y_abm_std**2, axis=2))  # Shape (n_steps, m)
        
        # Compute mean absolute error
        # Interpolate MFA to ABM time points
        y_mfa_interp = np.zeros_like(y_abm_mean_agg)
        for a in range(params['m']):
            y_mfa_interp[:, a] = np.interp(tau_abm, tau_mfa, y_mfa_agg[:, a])
        
        mae = np.mean(np.abs(y_mfa_interp - y_abm_mean_agg))
        
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Terminal MFA: y1={y_mfa_agg[-1, 0]:.3f}, y2={y_mfa_agg[-1, 1]:.3f}, y3={y_mfa_agg[-1, 2]:.3f}")
        print(f"Terminal ABM: y1={y_abm_mean_agg[-1, 0]:.3f}, y2={y_abm_mean_agg[-1, 1]:.3f}, y3={y_abm_mean_agg[-1, 2]:.3f}")
        
        results[regime] = {
            'tau_mfa': tau_mfa,
            'y_mfa': y_mfa_agg,
            'tau_abm': tau_abm,
            'y_abm_mean': y_abm_mean_agg,
            'y_abm_std': y_abm_std_agg,
            'y_abm_all': y_abm_all,
            'mae': mae,
            'terminal_mfa': y_mfa_agg[-1],
            'terminal_abm': y_abm_mean_agg[-1]
        }
    
    return results


def plot_results(results: Dict, output_dir: Path) -> None:
    """Plot validation results for all three regimes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    regimes = ['no_ranking', 'type_homophily', 'opinion_heterophily']
    regime_titles = {
        'no_ranking': 'No Ranking',
        'type_homophily': 'Type Homophily',
        'opinion_heterophily': 'Opinion Heterophily'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        res = results[regime]
        
        # Plot MFA trajectories
        for a in range(3):
            ax.plot(res['tau_mfa'], res['y_mfa'][:, a], 
                   label=f'MFA $y_{a+1}$', linewidth=2, linestyle='--')
        
        # Plot ABM mean trajectories with error bands
        for a in range(3):
            ax.plot(res['tau_abm'], res['y_abm_mean'][:, a],
                   label=f'ABM $y_{a+1}$', linewidth=2)
            ax.fill_between(
                res['tau_abm'],
                res['y_abm_mean'][:, a] - res['y_abm_std'][:, a],
                res['y_abm_mean'][:, a] + res['y_abm_std'][:, a],
                alpha=0.2
            )
        
        ax.set_xlabel(r'Time $\tau$', fontsize=12)
        ax.set_ylabel('Opinion fraction', fontsize=12)
        ax.set_title(f"{regime_titles[regime]}\nMAE={res['mae']:.4f}", fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exp1_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_dir / 'exp1_validation.png'}")


def save_metrics(results: Dict, output_dir: Path) -> None:
    """Save experiment metrics to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'exp1_metrics.txt', 'w') as f:
        f.write("EXPERIMENT 1: Mean-Field Approximation Validation\n")
        f.write("=" * 80 + "\n\n")
        
        for regime in ['no_ranking', 'type_homophily', 'opinion_heterophily']:
            res = results[regime]
            f.write(f"{regime.upper()}:\n")
            f.write(f"  Mean Absolute Error: {res['mae']:.6f}\n")
            f.write(f"  Terminal MFA: y1={res['terminal_mfa'][0]:.4f}, "
                   f"y2={res['terminal_mfa'][1]:.4f}, y3={res['terminal_mfa'][2]:.4f}\n")
            f.write(f"  Terminal ABM: y1={res['terminal_abm'][0]:.4f}, "
                   f"y2={res['terminal_abm'][1]:.4f}, y3={res['terminal_abm'][2]:.4f}\n\n")
    
    print(f"Metrics saved to {output_dir / 'exp1_metrics.txt'}")


if __name__ == "__main__":
    results = run_experiment_1()
    output_dir = Path("results/exp1")
    plot_results(results, output_dir)
    save_metrics(results, output_dir)
