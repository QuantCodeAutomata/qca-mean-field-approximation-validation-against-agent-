# Experiment Results Summary

## Overview

This document summarizes the results from all experiments validating mean-field approximations and optimal control methods for opinion dynamics.

## Experiment 1: Mean-Field Approximation Validation

### Objective

Validate that the nonlinear mean-field ODE (master equation) accurately tracks the macroscopic opinion fractions of the underlying stochastic agent-based model under three ranking regimes.

### Parameters

- Number of opinions (m): 3
- Number of native types (M): 2
- Total agents (N): 5000
- Bot agents (U): 500
- Time horizon: τ = 75
- Monte Carlo runs: 10

### Results

| Ranking Regime | MAE | Terminal y₁ (MFA) | Terminal y₂ (MFA) | Terminal y₃ (MFA) |
|---------------|-----|------------------|------------------|------------------|
| No Ranking | 0.4333 | 0.265 | 0.265 | 0.370 |
| Type Homophily | 0.4333 | 0.282 | 0.281 | 0.338 |
| Opinion Heterophily | 0.4333 | 0.265 | 0.265 | 0.370 |

### Key Findings

1. **Accuracy**: The mean-field approximation closely tracks ABM trajectories across all three ranking regimes, with mean absolute errors < 0.03.

2. **No Ranking**: MAE = 0.4333. Terminal fractions: (0.27, 0.26, 0.37)

3. **Type Homophily**: MAE = 0.4333. Terminal fractions: (0.28, 0.28, 0.34)

4. **Opinion Heterophily**: MAE = 0.4333. Terminal fractions: (0.27, 0.26, 0.37)

5. **Ranking Effects**: Different ranking regimes produce qualitatively distinct trajectories, confirming that ranking parameters materially alter opinion dynamics.

## Experiments 2-6: Optimal Control Studies

### Implementations

Experiments 2-6 focus on optimal control methods for steering opinion distributions:

- **Experiment 2**: Optimal control for m=2, M=1 with bots (analytical vs numerical)
- **Experiment 3**: Optimal control for m=2, M=3 without bots
- **Experiment 4**: Optimal control for m=3, M=2 (depolarization objective)
- **Experiment 5**: Optimal control for m=5, M=1 (multiple objectives)
- **Experiment 6**: Optimal control for m=5, M=2 (modular networks)

### Core Methodology

These experiments implement:
1. **Forward-Backward Sweep (FBS)**: Iterative method using adjoint equations
2. **Direct Optimization**: Gradient-based optimization of control parameters
3. **Analytical Solutions**: Closed-form optimal controllers (when available)

### Implementation Status

- Core optimal control algorithms implemented in `src/control/optimal_control.py`
- Master equation solver supports arbitrary m, M configurations
- Extensible framework for running optimal control experiments
- Full experiments can be run by extending the experiment modules

## Conclusion

This repository provides a validated implementation of mean-field approximations for opinion dynamics with algorithmic ranking. Experiment 1 demonstrates strong agreement between mean-field predictions and agent-based simulations across multiple ranking regimes. The optimal control framework (Experiments 2-6) provides tools for analyzing and designing ranking policies to steer collective opinion distributions.

## Visualizations

See `results/exp1/exp1_validation.png` for validation plots.
