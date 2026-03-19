"""Tests for transition probability tables."""

import numpy as np
import pytest
from src.core.transition_tables import (
    create_symmetric_assimilation_table_m2,
    create_m2_native_native_table,
    create_m2_native_bot_table,
    create_symmetric_assimilation_table_m3,
    create_symmetric_assimilation_table_m5,
    create_transition_tensor,
    verify_row_stochasticity
)


def test_row_stochasticity_m2():
    """Test that m=2 transition tables are row-stochastic."""
    P = create_symmetric_assimilation_table_m2(alpha=0.7)
    
    for s in range(2):
        for l in range(2):
            row_sum = np.sum(P[s, l, :])
            assert np.abs(row_sum - 1.0) < 1e-10, f"Row ({s},{l}) not stochastic: sum={row_sum}"


def test_row_stochasticity_m3():
    """Test that m=3 transition tables are row-stochastic."""
    P = create_symmetric_assimilation_table_m3(alpha=0.5, beta=0.3)
    
    for s in range(3):
        for l in range(3):
            row_sum = np.sum(P[s, l, :])
            assert np.abs(row_sum - 1.0) < 1e-10, f"Row ({s},{l}) not stochastic: sum={row_sum}"


def test_row_stochasticity_m5():
    """Test that m=5 transition tables are row-stochastic."""
    P = create_symmetric_assimilation_table_m5(alpha=0.5, beta=0.3)
    
    for s in range(5):
        for l in range(5):
            row_sum = np.sum(P[s, l, :])
            assert np.abs(row_sum - 1.0) < 1e-10, f"Row ({s},{l}) not stochastic: sum={row_sum}"


def test_m2_native_native_table():
    """Test m=2 native-native transition table from paper."""
    P = create_m2_native_native_table()
    
    # Check specific values from Eq. 32
    assert P[0, 0, 0] == 1.0
    assert P[0, 0, 1] == 0.0
    assert P[0, 1, 0] == 0.5
    assert P[0, 1, 1] == 0.5
    assert P[1, 0, 0] == 0.5
    assert P[1, 0, 1] == 0.5
    assert P[1, 1, 0] == 0.0
    assert P[1, 1, 1] == 1.0
    
    # Check row-stochasticity
    for s in range(2):
        for l in range(2):
            assert np.abs(np.sum(P[s, l, :]) - 1.0) < 1e-10


def test_m2_native_bot_table():
    """Test m=2 native-bot transition table from paper."""
    P = create_m2_native_bot_table()
    
    # Check specific values from Eq. 33
    assert P[0, 1, 0] == 0.3
    assert P[0, 1, 1] == 0.7
    assert P[1, 1, 0] == 0.0
    assert P[1, 1, 1] == 1.0
    
    # Check row-stochasticity
    for s in range(2):
        for l in range(2):
            assert np.abs(np.sum(P[s, l, :]) - 1.0) < 1e-10


def test_symmetric_assimilation_properties():
    """Test properties of symmetric assimilation tables."""
    alpha = 0.6
    P = create_symmetric_assimilation_table_m2(alpha)
    
    # When s==l, should stay with higher probability
    for a in range(2):
        assert P[a, a, a] >= alpha


def test_transition_tensor_shape():
    """Test that transition tensor has correct shape."""
    m, M = 3, 2
    P = create_transition_tensor(m, M, include_bots=True)
    
    assert P.shape == (m, m, m, M+1, M+1)
    
    P_no_bots = create_transition_tensor(m, M, include_bots=False)
    assert P_no_bots.shape == (m, m, m, M, M)


def test_transition_tensor_stochasticity():
    """Test that full transition tensor is row-stochastic."""
    m, M = 3, 2
    P = create_transition_tensor(m, M, include_bots=True)
    
    assert verify_row_stochasticity(P), "Transition tensor is not row-stochastic"


def test_different_table_types():
    """Test that different table types can be created."""
    m, M = 2, 2
    
    P_sym = create_transition_tensor(m, M, include_bots=False, table_type="symmetric")
    assert P_sym.shape == (m, m, m, M, M)
    
    P_native = create_transition_tensor(m, M, include_bots=False, table_type="m2_native")
    assert P_native.shape == (m, m, m, M, M)
    
    P_bot = create_transition_tensor(m, M, include_bots=False, table_type="m2_bot")
    assert P_bot.shape == (m, m, m, M, M)


def test_probability_bounds():
    """Test that all probabilities are in [0, 1]."""
    m, M = 5, 2
    P = create_transition_tensor(m, M, include_bots=True)
    
    assert np.all(P >= 0.0), "Negative probabilities found"
    assert np.all(P <= 1.0), "Probabilities exceeding 1 found"


def test_alpha_beta_parameters():
    """Test that alpha and beta parameters affect the table."""
    P1 = create_symmetric_assimilation_table_m3(alpha=0.5, beta=0.3)
    P2 = create_symmetric_assimilation_table_m3(alpha=0.7, beta=0.2)
    
    assert not np.allclose(P1, P2), "Different parameters should produce different tables"


def test_m5_table_intermediate_opinions():
    """Test that m=5 table distributes mass to intermediate opinions."""
    P = create_symmetric_assimilation_table_m5(alpha=0.4, beta=0.2)
    
    # For s=0, l=4 (extreme opinions), check intermediate opinions get non-zero mass
    intermediate_mass = P[0, 4, 1] + P[0, 4, 2] + P[0, 4, 3]
    expected_mass = 1.0 - 0.4 - 0.2  # 1 - alpha - beta
    assert np.abs(intermediate_mass - expected_mass) < 1e-10


def test_verify_row_stochasticity_function():
    """Test the verification function itself."""
    m, M = 2, 2
    
    # Valid table
    P_valid = create_transition_tensor(m, M, include_bots=False)
    assert verify_row_stochasticity(P_valid)
    
    # Invalid table (manually break stochasticity)
    P_invalid = P_valid.copy()
    P_invalid[0, 0, :, 0, 0] = np.array([0.6, 0.6])  # Sum to 1.2
    assert not verify_row_stochasticity(P_invalid, tol=1e-5)


def test_edge_case_alpha_beta():
    """Test edge cases for alpha and beta parameters."""
    # alpha + beta should not exceed 1
    P = create_symmetric_assimilation_table_m3(alpha=0.7, beta=0.3)
    
    for s in range(3):
        for l in range(3):
            row_sum = np.sum(P[s, l, :])
            assert np.abs(row_sum - 1.0) < 1e-10


def test_same_opinion_dynamics():
    """Test that same opinion pairs preserve the opinion."""
    P = create_symmetric_assimilation_table_m5(alpha=0.5, beta=0.3)
    
    # When both hold same opinion, should stay with probability 1
    for a in range(5):
        assert np.abs(P[a, a, a] - 1.0) < 1e-10, f"Same opinion ({a},{a}) should preserve with p=1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
