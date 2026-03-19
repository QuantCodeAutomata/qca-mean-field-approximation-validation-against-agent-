"""
Transition probability tables for opinion dynamics.

This module provides functions to create transition probability tables P^{f,r}
as specified in the paper's Appendix A.3.
"""

import numpy as np


def create_symmetric_assimilation_table_m2(alpha: float = 0.7) -> np.ndarray:
    """
    Create symmetric assimilative transition table for m=2 opinions.
    
    For m=2, the table structure is:
    - p_{s,l,k} = alpha if k == l (adopt source opinion)
    - p_{s,l,k} = (1-alpha) if k != l (maintain object opinion)
    
    Parameters
    ----------
    alpha : float
        Assimilation strength (0.5 < alpha < 1)
    
    Returns
    -------
    P : np.ndarray
        Transition table, shape (2, 2, 2)
        P[s, l, k] = probability to adopt k given object s and source l
    """
    P = np.zeros((2, 2, 2))
    
    for s in range(2):
        for l in range(2):
            P[s, l, l] = alpha
            P[s, l, 1 - l] = 1.0 - alpha
    
    return P


def create_m2_native_native_table() -> np.ndarray:
    """
    Create transition table for m=2 native-native interactions (Eq. 32 from paper).
    
    Returns
    -------
    P : np.ndarray
        Transition table, shape (2, 2, 2)
    """
    P = np.zeros((2, 2, 2))
    
    # p_{1,1,1} = 1, p_{1,1,2} = 0
    P[0, 0, 0] = 1.0
    P[0, 0, 1] = 0.0
    
    # p_{1,2,1} = 0.5, p_{1,2,2} = 0.5
    P[0, 1, 0] = 0.5
    P[0, 1, 1] = 0.5
    
    # p_{2,1,1} = 0.5, p_{2,1,2} = 0.5
    P[1, 0, 0] = 0.5
    P[1, 0, 1] = 0.5
    
    # p_{2,2,1} = 0, p_{2,2,2} = 1
    P[1, 1, 0] = 0.0
    P[1, 1, 1] = 1.0
    
    return P


def create_m2_native_bot_table() -> np.ndarray:
    """
    Create transition table for m=2 native-bot interactions (Eq. 33 from paper).
    
    Bots push toward opinion Z_2.
    
    Returns
    -------
    P : np.ndarray
        Transition table, shape (2, 2, 2)
    """
    P = np.zeros((2, 2, 2))
    
    # p_{1,2,1} = 0.3, p_{1,2,2} = 0.7 (bots hold opinion 2)
    P[0, 1, 0] = 0.3
    P[0, 1, 1] = 0.7
    
    # p_{2,2,1} = 0, p_{2,2,2} = 1
    P[1, 1, 0] = 0.0
    P[1, 1, 1] = 1.0
    
    # When bot holds opinion 1 (not used in standard setup)
    P[0, 0, 0] = 0.7
    P[0, 0, 1] = 0.3
    P[1, 0, 0] = 1.0
    P[1, 0, 1] = 0.0
    
    return P


def create_symmetric_assimilation_table_m3(alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
    """
    Create symmetric assimilative transition table for m=3 opinions.
    
    Parameters
    ----------
    alpha : float
        Probability to adopt source opinion
    beta : float
        Probability to maintain object opinion
    
    Returns
    -------
    P : np.ndarray
        Transition table, shape (3, 3, 3)
    """
    P = np.zeros((3, 3, 3))
    
    for s in range(3):
        for l in range(3):
            if s == l:
                # Same opinion: stay with probability 1
                P[s, l, s] = 1.0
            else:
                # Different opinions
                # Adopt source opinion with probability alpha
                P[s, l, l] = alpha
                # Maintain object opinion with probability beta
                P[s, l, s] = beta
                # Remaining probability distributed to other opinion
                remaining = 1.0 - alpha - beta
                for k in range(3):
                    if k != s and k != l:
                        P[s, l, k] = remaining
    
    return P


def create_symmetric_assimilation_table_m5(alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
    """
    Create symmetric assimilative transition table for m=5 opinions.
    
    Uses compromise dynamics where opinions can shift toward source opinion.
    
    Parameters
    ----------
    alpha : float
        Probability to adopt source opinion
    beta : float
        Probability to maintain object opinion
    
    Returns
    -------
    P : np.ndarray
        Transition table, shape (5, 5, 5)
    """
    P = np.zeros((5, 5, 5))
    
    for s in range(5):
        for l in range(5):
            if s == l:
                # Same opinion: stay with high probability
                P[s, l, s] = 1.0
            else:
                # Different opinions: compromise dynamics
                P[s, l, l] = alpha  # Adopt source
                P[s, l, s] = beta   # Maintain object
                
                # Remaining mass distributed to intermediate opinions
                remaining = 1.0 - alpha - beta
                n_intermediate = 3  # m - 2
                
                if n_intermediate > 0:
                    for k in range(5):
                        if k != s and k != l:
                            P[s, l, k] = remaining / n_intermediate
    
    return P


def create_transition_tensor(
    m: int,
    M: int,
    include_bots: bool = True,
    table_type: str = "symmetric",
    alpha: float = 0.5,
    beta: float = 0.3
) -> np.ndarray:
    """
    Create full transition probability tensor for all (f,r) pairs.
    
    Parameters
    ----------
    m : int
        Number of opinions
    M : int
        Number of native types
    include_bots : bool
        Include bot type (M+1 total types)
    table_type : str
        Type of transition table: "symmetric", "m2_native", "m2_bot"
    alpha : float
        Assimilation parameter
    beta : float
        Persistence parameter
    
    Returns
    -------
    P : np.ndarray
        Transition tensor, shape (m, m, m, M+1, M+1) if include_bots,
        else (m, m, m, M, M)
    """
    n_types = M + 1 if include_bots else M
    P = np.zeros((m, m, m, n_types, n_types))
    
    # Create base table
    if table_type == "symmetric":
        if m == 2:
            base_table = create_symmetric_assimilation_table_m2(alpha)
        elif m == 3:
            base_table = create_symmetric_assimilation_table_m3(alpha, beta)
        elif m == 5:
            base_table = create_symmetric_assimilation_table_m5(alpha, beta)
        else:
            raise ValueError(f"Unsupported m={m} for symmetric table")
    elif table_type == "m2_native":
        base_table = create_m2_native_native_table()
    elif table_type == "m2_bot":
        base_table = create_m2_native_bot_table()
    else:
        raise ValueError(f"Unknown table_type: {table_type}")
    
    # Replicate for all (f, r) pairs (native-native)
    for f in range(M):
        for r in range(M):
            P[:, :, :, f, r] = base_table
    
    # Native-bot interactions
    if include_bots:
        if table_type == "m2_bot":
            bot_table = base_table
        else:
            bot_table = base_table  # Use same table for bots
        
        for f in range(M):
            P[:, :, :, f, M] = bot_table
            P[:, :, :, M, f] = bot_table
        P[:, :, :, M, M] = bot_table
    
    return P


def verify_row_stochasticity(P: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Verify that transition table is row-stochastic.
    
    Parameters
    ----------
    P : np.ndarray
        Transition tensor
    tol : float
        Tolerance for sum check
    
    Returns
    -------
    is_valid : bool
        True if all rows sum to 1
    """
    m, m2, m3, n_f, n_r = P.shape
    assert m == m2 == m3, "First three dimensions must match"
    
    for f in range(n_f):
        for r in range(n_r):
            for s in range(m):
                for l in range(m):
                    row_sum = np.sum(P[s, l, :, f, r])
                    if abs(row_sum - 1.0) > tol:
                        print(f"Row ({s},{l},{f},{r}) sums to {row_sum}")
                        return False
    
    return True
