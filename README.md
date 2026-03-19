# Mean-Field Approximation Validation Against Agent-Based Simulations

This repository implements experiments validating that nonlinear mean-field ordinary differential equations (master equations) accurately track the macroscopic opinion fractions of underlying stochastic agent-based models (ABM) in opinion dynamics.

## Overview

The experiments reproduce results from research on opinion dynamics under algorithmic ranking, validating:
- Mean-field approximation accuracy across three ranking regimes (no ranking, type homophily, opinion heterophily)
- Optimal control methods for steering opinion distributions
- Analytical vs. numerical optimization approaches (Forward-Backward Sweep, Direct methods)

## Project Structure

```
.
├── src/
│   ├── core/           # Core simulation and ODE solvers
│   ├── experiments/    # Individual experiment implementations
│   └── control/        # Optimal control methods
├── tests/              # Test suite
├── results/            # Experiment outputs (metrics, plots)
└── README.md
```

## Experiments

1. **Experiment 1**: Mean-field validation for m=3 opinions, M=2 types, N=5000 agents
2. **Experiment 2**: Optimal control for m=2, M=1 with bots (analytical vs numerical)
3. **Experiment 3**: Optimal control for m=2, M=3 without bots (activity rate variations)
4. **Experiment 4**: Optimal control for m=3, M=2 (correlated vs uncorrelated states)
5. **Experiment 5**: Optimal control for m=5, M=1 (depolarization and nudging)
6. **Experiment 6**: Optimal control for m=5, M=2 (steering left with modular network)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:
```bash
python -m src.experiments.run_all_experiments
```

Run individual experiments:
```bash
python -m src.experiments.exp_1_validation
```

Run tests:
```bash
pytest tests/
```

## Results

All results including metrics and visualizations are saved to the `results/` directory.
Summary metrics are available in `results/RESULTS.md`.

## License

MIT License
