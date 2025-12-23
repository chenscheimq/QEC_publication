"""
QEC Utilities Module

Contains utility functions for quantum error correction simulations:
- Unit conversion helpers
- Plotting utilities
- Thermal noise model (collapse operators for open-system dynamics)
- Spectral density functions

Usage:
    from qec_config.utils import c_ops_gen_thermal, bose_einstein_N, break_at_swaps
"""

import numpy as np
from qutip import tensor, sigmam, qeye

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant [J/K]
HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]


# =============================================================================
# Unit Conversion Helpers
# =============================================================================

def MHz_to_rad_s(freq_MHz):
    """Convert frequency from MHz to rad/s."""
    return freq_MHz * 2 * np.pi * 1e6


def rad_s_to_MHz(omega_rad_s):
    """Convert angular frequency from rad/s to MHz."""
    return omega_rad_s / (2 * np.pi * 1e6)


def us_to_s(time_us):
    """Convert time from microseconds to seconds."""
    return time_us * 1e-6


def s_to_us(time_s):
    """Convert time from seconds to microseconds."""
    return time_s * 1e6


# =============================================================================
# Plotting Utilities
# =============================================================================

def break_at_swaps(y_vals, idx_series):
    """
    Insert NaN at eigenstate assignment changes to break plot lines.

    When tracking energy eigenvalues through an adiabatic protocol, the
    eigenstate indices can swap at avoided crossings. This function inserts
    NaN values at those swap points so that matplotlib draws discontinuous
    lines instead of connecting unrelated states.

    Parameters
    ----------
    y_vals : array-like
        Array of y-values (typically energies) to process
    idx_series : array-like
        Array of eigenstate indices corresponding to each y-value

    Returns
    -------
    ndarray
        Copy of y_vals with NaN inserted at swap points

    Example
    -------
    >>> energies_broken = break_at_swaps(energies[0] * TO_FREQ_UNITS, idx_series[0])
    >>> ax.plot(t, energies_broken)  # Will have gaps at avoided crossings
    """
    y_broken = np.array(y_vals, dtype=float).copy()
    idx_arr = np.array(idx_series, dtype=int)
    swaps = np.where(idx_arr[1:] != idx_arr[:-1])[0] + 1
    for s in swaps:
        if 0 < s < len(y_broken):
            y_broken[s] = np.nan
    return y_broken


# =============================================================================
# Thermal Noise Model
# =============================================================================

def bose_einstein_N(omega, T_K):
    """
    Bose-Einstein occupation number for a mode at frequency omega and temperature T.

    Parameters
    ----------
    omega : float
        Angular frequency in rad/s (must be positive)
    T_K : float
        Temperature in Kelvin

    Returns
    -------
    float
        Mean thermal occupation number n_th = 1/(exp(ℏω/k_B T) - 1)
        Returns 0 if T_K <= 0 or omega <= 0
    """
    if T_K <= 0 or omega <= 0:
        return 0.0

    exponent = (HBAR * omega) / (K_B * T_K)

    # Avoid overflow for large exponents
    if exponent > 700:
        return 0.0

    return 1.0 / (np.exp(exponent) - 1.0)


def spectral_amp(omega, lambda_2):
    """
    Spectral amplitude for flat (Ohmic) spectral density.

    Parameters
    ----------
    omega : float
        Angular frequency in rad/s (must be positive)
    lambda_2 : float
        Coupling strength parameter (rate prefactor)

    Returns
    -------
    float
        Square root of spectral density: sqrt(lambda_2) for omega > 0, else 0
    """
    if omega <= 0:
        return 0.0
    return np.sqrt(lambda_2)


def c_ops_gen_thermal(t_eval, H_func, n_qubits, lambda_2, T_K):
    """
    Generate thermal collapse operators at a specific time for open-system dynamics.

    This function computes instantaneous Lindblad collapse operators based on
    the eigenstates of the Hamiltonian at time t_eval. It models thermal
    transitions between the code space (ground states) and error states due
    to coupling to a thermal bath.

    The model assumes:
    - Flat (Ohmic) spectral density
    - Single-qubit σ⁻ coupling to the bath
    - Secular approximation (only energy-conserving transitions)

    Parameters
    ----------
    t_eval : float
        Time at which to evaluate the Hamiltonian (in seconds)
    H_func : callable
        Function H_func(t, args) -> Qobj that returns the Hamiltonian at time t.
        The 'args' parameter can be None.
    n_qubits : int
        Number of qubits in the system
    lambda_2 : float
        Coupling strength parameter for the spectral density
    T_K : float
        Temperature of the thermal bath in Kelvin

    Returns
    -------
    list of Qobj
        List of collapse operators for use with qutip.mesolve

    Example
    -------
    >>> # Piecewise master equation simulation
    >>> rho = initial_state * initial_state.dag()
    >>> for k in range(1, len(t_list)):
    ...     t_mid = 0.5 * (t_list[k-1] + t_list[k])
    ...     c_ops = c_ops_gen_thermal(t_mid, H_func, n_qubits, lambda_2, T_K)
    ...     result = mesolve(H_func, rho, [t_list[k-1], t_list[k]], c_ops=c_ops)
    ...     rho = result.states[-1]

    Notes
    -----
    - Transitions are computed from the ground state (index 0) to excited states
    - Both absorption (heating) and emission (cooling) processes are included
    - The rates follow detailed balance: Γ_abs/Γ_em = n_th/(n_th + 1)
    """
    # Get instantaneous Hamiltonian eigenstates
    Ht = H_func(t_eval, None)
    evals, evecs = Ht.eigenstates()
    dim = len(evals)

    # Identity operator for tensor products
    I = qeye(2)

    c_ops = []

    # Consider transitions from ground state (0) to excited states (b >= 2)
    # States 0 and 1 are assumed to be the logical code space
    for b in range(2, dim):
        omega = float(evals[b] - evals[0])  # Transition frequency

        if abs(omega) < 1e-8:
            continue  # Skip degenerate transitions

        rate_abs = 0.0  # Absorption rate: |0⟩ → |b⟩
        rate_em = 0.0   # Emission rate: |b⟩ → |0⟩

        # Sum contributions from each qubit's σ⁻ operator
        for i in range(n_qubits):
            # Build σ⁻ operator for qubit i
            sigmam_i = tensor([sigmam() if j == i else I for j in range(n_qubits)])

            # Matrix element ⟨b|σ⁻ᵢ|0⟩
            m_ba = evecs[b].dag() * sigmam_i * evecs[0]
            mag2 = float(np.abs(m_ba)**2)

            if mag2 < 1e-8:
                continue

            # Spectral amplitude and thermal occupation
            g = spectral_amp(abs(omega), lambda_2)
            N = bose_einstein_N(abs(omega), T_K)

            # Accumulate rates (Fermi's golden rule)
            rate_abs += N * (g**2) * mag2
            rate_em += (N + 1) * (g**2) * mag2

        # Create collapse operators for non-negligible rates
        if rate_abs > 1e-8:
            # Absorption: |0⟩ → |b⟩
            Lop = (evecs[b] * evecs[0].dag()).to('csr')
            c_ops.append(np.sqrt(rate_abs) * Lop)

        if rate_em > 1e-8:
            # Emission: |b⟩ → |0⟩
            Lop = (evecs[0] * evecs[b].dag()).to('csr')
            c_ops.append(np.sqrt(rate_em) * Lop)

    return c_ops


def c_ops_gen_thermal_full(t_eval, H_func, n_qubits, lambda_2, T_K):
    """
    Generate thermal collapse operators considering all state transitions.

    Extended version of c_ops_gen_thermal that includes transitions between
    all pairs of eigenstates, not just from the ground state.

    Parameters
    ----------
    t_eval : float
        Time at which to evaluate the Hamiltonian (in seconds)
    H_func : callable
        Function H_func(t, args) -> Qobj that returns the Hamiltonian at time t
    n_qubits : int
        Number of qubits in the system
    lambda_2 : float
        Coupling strength parameter for the spectral density
    T_K : float
        Temperature of the thermal bath in Kelvin

    Returns
    -------
    list of Qobj
        List of collapse operators for use with qutip.mesolve
    """
    Ht = H_func(t_eval, None)
    evals, evecs = Ht.eigenstates()
    dim = len(evals)

    I = qeye(2)
    c_ops = []

    # Consider all pairs of states
    for a in range(dim):
        for b in range(a + 1, dim):
            omega = float(evals[b] - evals[a])  # E_b > E_a

            if abs(omega) < 1e-8:
                continue

            rate_abs = 0.0  # |a⟩ → |b⟩ (absorption)
            rate_em = 0.0   # |b⟩ → |a⟩ (emission)

            for i in range(n_qubits):
                sigmam_i = tensor([sigmam() if j == i else I for j in range(n_qubits)])
                sigmap_i = sigmam_i.dag()

                # σ⁻ matrix element for emission
                m_ab_minus = evecs[a].dag() * sigmam_i * evecs[b]
                mag2_minus = float(np.abs(m_ab_minus)**2)

                # σ⁺ matrix element for absorption
                m_ba_plus = evecs[b].dag() * sigmap_i * evecs[a]
                mag2_plus = float(np.abs(m_ba_plus)**2)

                g = spectral_amp(abs(omega), lambda_2)
                N = bose_einstein_N(abs(omega), T_K)

                if mag2_plus > 1e-8:
                    rate_abs += N * (g**2) * mag2_plus
                if mag2_minus > 1e-8:
                    rate_em += (N + 1) * (g**2) * mag2_minus

            if rate_abs > 1e-8:
                Lop = (evecs[b] * evecs[a].dag()).to('csr')
                c_ops.append(np.sqrt(rate_abs) * Lop)

            if rate_em > 1e-8:
                Lop = (evecs[a] * evecs[b].dag()).to('csr')
                c_ops.append(np.sqrt(rate_em) * Lop)

    return c_ops
