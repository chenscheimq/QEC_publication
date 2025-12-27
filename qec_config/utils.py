"""
QEC Utilities Module

Contains utility functions for quantum error correction simulations:
- Unit conversion helpers
- Plotting utilities
- Thermal noise model (collapse operators for open-system dynamics)
- Spectral density functions
- Fidelity simulation helpers

Usage:
    from qec_config.utils import c_ops_gen_thermal, bose_einstein_N, break_at_swaps
    from qec_config.utils import simulate_rap_fidelity
"""

import numpy as np
from qutip import tensor, sigmam, qeye, mesolve

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
    # H_func can be either H(t) or H(t, args)
    try:
        Ht = H_func(t_eval)
    except TypeError:
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
        Function H_func(t) -> Qobj that returns the Hamiltonian at time t
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
    try:
        Ht = H_func(t_eval)
    except TypeError:
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


# =============================================================================
# Ornstein-Uhlenbeck (1/f-like) Control Noise
# =============================================================================

def sigma_from_rms(target_rms, tau):
    """
    Convert target RMS amplitude to Ornstein-Uhlenbeck noise strength sigma.

    The stationary variance of an OU process is σ²τ/2, so RMS = σ√(τ/2).
    Inverting: σ = √2 × RMS / √τ

    Parameters
    ----------
    target_rms : float
        Desired RMS amplitude of the noise (in same units as sigma, e.g., rad/s)
    tau : float
        Correlation time of the OU process (in seconds)

    Returns
    -------
    float
        OU noise strength parameter sigma

    Example
    -------
    >>> omega_max = 2*np.pi*25e6  # 25 MHz
    >>> rms = 0.2 * omega_max     # 20% RMS
    >>> tau = 0.6e-6              # 0.6 µs correlation time
    >>> sigma = sigma_from_rms(rms, tau)
    """
    return (np.sqrt(2.0) * float(target_rms)) / np.sqrt(float(tau))


def rms_from_sigma(sigma, tau):
    """
    Convert OU noise strength sigma to RMS amplitude.

    Inverse of sigma_from_rms: RMS = σ√(τ/2)

    Parameters
    ----------
    sigma : float
        OU noise strength parameter
    tau : float
        Correlation time of the OU process (in seconds)

    Returns
    -------
    float
        RMS amplitude of the noise
    """
    return float(sigma) * np.sqrt(float(tau) / 2.0)


def ou_trace(times, tau, sigma, rng=None, seed=None):
    """
    Generate an Ornstein-Uhlenbeck (OU) process trace.

    The OU process models 1/f-like colored noise with exponential autocorrelation.
    It is used to simulate local control noise in quantum systems.

    The discretized update equation is:
        x(t+dt) = x(t) - (x(t)/τ)dt + σ√dt × ξ

    where ξ is standard Gaussian white noise.

    Parameters
    ----------
    times : array-like
        Time points at which to evaluate the process (in seconds)
    tau : float
        Correlation time (decorrelation timescale) in seconds
    sigma : float
        Noise strength parameter (use sigma_from_rms to convert from RMS)
    rng : numpy.random.Generator, optional
        Random number generator. If None and seed is None, creates new generator.
    seed : int, optional
        Random seed for reproducibility. Ignored if rng is provided.

    Returns
    -------
    ndarray
        OU process values at each time point

    Example
    -------
    >>> t_list = np.linspace(0, 4e-6, 1001)
    >>> tau = 0.6e-6  # 0.6 µs
    >>> sigma = sigma_from_rms(0.2 * 2*np.pi*25e6, tau)
    >>> noise = ou_trace(t_list, tau, sigma, seed=42)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    times = np.asarray(times, dtype=float)
    dt = np.diff(times, prepend=times[0])

    x = 0.0
    out = np.empty_like(times, dtype=float)

    for i, dti in enumerate(dt):
        x += (-x / tau) * dti + sigma * np.sqrt(max(dti, 1e-16)) * rng.normal()
        out[i] = x

    return out


def generate_noise_traces(t_list, tau_X, sigma_X, tau_Z, sigma_Z, n_qubits, seed=None):
    """
    Generate OU noise traces for all qubits.

    Parameters
    ----------
    t_list : array-like
        Time points
    tau_X : float
        Correlation time for transverse (X) noise
    sigma_X : float
        Noise strength for transverse noise
    tau_Z : float
        Correlation time for longitudinal (Z) noise
    sigma_Z : float
        Noise strength for longitudinal noise
    n_qubits : int
        Number of qubits (3 for rep code, 4 for Bacon-Shor)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X_traces, Z_traces) where each is a list of n_qubits noise arrays
    """
    rng = np.random.default_rng(seed)

    X_traces = [ou_trace(t_list, tau_X, sigma_X, rng) for _ in range(n_qubits)]
    Z_traces = [ou_trace(t_list, tau_Z, sigma_Z, rng) for _ in range(n_qubits)]

    return X_traces, Z_traces


def simulate_control_noise_fidelity(config, Ep, rms_X_frac, tau_X, rms_Z_frac=0.0, tau_Z=None,
                                     n_realizations=10, n_steps=None, seed=None, verbose=False):
    """
    Simulate RAP protocol with local OU (1/f-like) control noise.

    This models noise in the control fields (drive amplitude and detuning) as
    Ornstein-Uhlenbeck processes with specified RMS and correlation time.

    Parameters
    ----------
    config : QECConfigBase
        QEC configuration object (QECConfig or BaconShorConfig)
    Ep : float
        Penalty energy in rad/s
    rms_X_frac : float
        RMS of transverse (X) noise as fraction of omega_max
    tau_X : float
        Correlation time for X noise in seconds
    rms_Z_frac : float, optional
        RMS of longitudinal (Z) noise as fraction of omega_max (default 0)
    tau_Z : float, optional
        Correlation time for Z noise in seconds (default: same as tau_X)
    n_realizations : int, optional
        Number of noise realizations to average over (default 10)
    n_steps : int, optional
        Number of time steps for simulation
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        If True, print progress

    Returns
    -------
    dict
        Dictionary containing:
        - 'fidelity_mean': mean fidelity over realizations
        - 'fidelity_std': standard deviation
        - 'fidelity_logical_mean': mean logical fidelity
        - 'fidelity_logical_std': std of logical fidelity
        - 'fidelities': array of individual fidelities
        - 'fidelities_logical': array of individual logical fidelities
    """
    from qutip import sesolve, Qobj

    if tau_Z is None:
        tau_Z = tau_X

    # Determine if Bacon-Shor
    is_bacon_shor = hasattr(config, 'code_name') and config.code_name == 'bacon_shor'
    n_qubits = 4 if is_bacon_shor else 3

    # Get parameters from config
    omega_max = config.omega_max
    T_max = config.T_max

    # Convert RMS fractions to sigma values
    rms_X = rms_X_frac * omega_max
    rms_Z = rms_Z_frac * omega_max
    sigma_X = sigma_from_rms(rms_X, tau_X)
    sigma_Z = sigma_from_rms(rms_Z, tau_Z) if rms_Z_frac > 0 else 0.0

    # Get time list
    if n_steps is not None and n_steps < len(config.t_list):
        indices = np.linspace(0, len(config.t_list) - 1, n_steps, dtype=int)
        t_list = config.t_list[indices]
    else:
        t_list = config.t_list

    # Get operators
    X_L = config.X_L
    Z_L = config.Z_L
    S_total = config.get_stabilizer_penalty()

    # Single-qubit X and Z operators
    X_ops = config.get_single_qubit_X_ops()
    Z_ops = config.get_single_qubit_Z_ops()

    # Clean pulse functions
    def omega_t(t):
        return config.omega_func(t)

    def delta_t(t):
        return config.delta_func(t)

    # Projectors for fidelity
    proj_L0 = config.logical_zero * config.logical_zero.dag()
    proj_L1 = config.logical_one * config.logical_one.dag()

    if is_bacon_shor:
        proj_A0 = config.ancilla_zero * config.ancilla_zero.dag()
        proj_A1 = config.ancilla_one * config.ancilla_one.dag()
        e_ops = [proj_L0, proj_L1, proj_A0, proj_A1]
    else:
        e_ops = [proj_L0, proj_L1]

    # Initial state
    psi0 = config.logical_zero

    # Master RNG
    rng_master = np.random.default_rng(seed)

    fidelities = []
    fidelities_logical = []

    for r in range(n_realizations):
        # Generate noise traces for this realization
        r_seed = rng_master.integers(1 << 30)
        X_traces, Z_traces = generate_noise_traces(
            t_list, tau_X, sigma_X, tau_Z, sigma_Z, n_qubits, seed=r_seed
        )

        # Create interpolation functions for noise
        from scipy.interpolate import interp1d

        xi_X_funcs = [interp1d(t_list, X_traces[i], kind='linear', fill_value='extrapolate')
                      for i in range(n_qubits)]
        xi_Z_funcs = [interp1d(t_list, Z_traces[i], kind='linear', fill_value='extrapolate')
                      for i in range(n_qubits)]

        # Build time-dependent Hamiltonian
        def make_coeff(func):
            return lambda t, args: float(func(t))

        H = [
            [X_L, lambda t, args: omega_t(t)],
            [Z_L, lambda t, args: delta_t(t)],
            -Ep * S_total,
        ]

        # Add noise terms
        for i in range(n_qubits):
            H.append([X_ops[i], make_coeff(xi_X_funcs[i])])
            if sigma_Z > 0:
                H.append([Z_ops[i], make_coeff(xi_Z_funcs[i])])

        # Solve
        try:
            result = sesolve(H, psi0, t_list, e_ops=e_ops)
            fid_L1 = float(np.real(result.expect[1][-1]))
            if is_bacon_shor:
                fid_A1 = float(np.real(result.expect[3][-1]))
                fid_total = fid_L1 + fid_A1
            else:
                fid_total = fid_L1
        except Exception as e:
            if verbose:
                print(f"  Realization {r} failed: {e}")
            fid_total = 0.0
            fid_L1 = 0.0

        fidelities.append(fid_total)
        fidelities_logical.append(fid_L1)

        if verbose and (r + 1) % 10 == 0:
            print(f"  Realization {r+1}/{n_realizations}, fid = {fid_total:.4f}")

    fidelities = np.array(fidelities)
    fidelities_logical = np.array(fidelities_logical)

    return {
        'fidelity_mean': fidelities.mean(),
        'fidelity_std': fidelities.std(),
        'fidelity_logical_mean': fidelities_logical.mean(),
        'fidelity_logical_std': fidelities_logical.std(),
        'fidelities': fidelities,
        'fidelities_logical': fidelities_logical,
    }


def compute_Ep_vs_rms_curve(config, tau_X, rms_fracs, Ep_grid, target_fid=0.99,
                             tau_Z=None, rms_Z_frac=0.0, n_realizations=5,
                             n_steps=51, seed=None, verbose=True):
    """
    Compute required Ep to achieve target fidelity for each RMS amplitude.

    For each RMS value, finds the minimum Ep from Ep_grid that achieves
    the target fidelity.

    Parameters
    ----------
    config : QECConfigBase
        QEC configuration object
    tau_X : float
        Correlation time for X noise in seconds
    rms_fracs : array-like
        Array of RMS fractions (of omega_max) to sweep
    Ep_grid : array-like
        Grid of Ep values to search (in rad/s)
    target_fid : float, optional
        Target fidelity (default 0.99)
    tau_Z : float, optional
        Correlation time for Z noise
    rms_Z_frac : float, optional
        Fixed Z noise RMS fraction
    n_realizations : int, optional
        Number of noise realizations per (Ep, RMS) point
    n_steps : int, optional
        Number of time steps
    seed : int, optional
        Random seed
    verbose : bool, optional
        If True, print progress

    Returns
    -------
    dict
        Dictionary containing:
        - 'Ep_required': array of required Ep values (NaN if not achieved)
        - 'rms_fracs': input RMS fractions
        - 'rms_MHz': RMS values in MHz
        - 'tau_X': correlation time used
        - 'target_fid': target fidelity used
    """
    omega_max = config.omega_max
    rms_fracs = np.asarray(rms_fracs)
    Ep_grid = np.asarray(Ep_grid)

    Ep_required = np.full(len(rms_fracs), np.nan)

    rng_master = np.random.default_rng(seed)

    for i, rms_frac in enumerate(rms_fracs):
        if verbose:
            print(f"  RMS = {rms_frac*100:.1f}% ...", end="", flush=True)

        # Search Ep grid
        for Ep in Ep_grid:
            r_seed = rng_master.integers(1 << 30)
            result = simulate_control_noise_fidelity(
                config, Ep, rms_frac, tau_X,
                rms_Z_frac=rms_Z_frac, tau_Z=tau_Z,
                n_realizations=n_realizations, n_steps=n_steps,
                seed=r_seed, verbose=False
            )

            if result['fidelity_mean'] >= target_fid:
                Ep_required[i] = Ep
                break

        if verbose:
            if np.isnan(Ep_required[i]):
                print(" not achieved")
            else:
                print(f" Ep = {Ep_required[i]/(2*np.pi*1e6):.1f} MHz")

    return {
        'Ep_required': Ep_required,
        'rms_fracs': rms_fracs,
        'rms_MHz': rms_fracs * omega_max / (2 * np.pi * 1e6),
        'tau_X': tau_X,
        'target_fid': target_fid,
    }


# =============================================================================
# Fidelity Simulation Helpers
# =============================================================================

def simulate_rap_fidelity(config, initial_state, target_state, Ep, T_K, lambda_2,
                          n_steps=None, return_trajectory=False, verbose=False):
    """
    Simulate RAP protocol with thermal noise and compute fidelity.

    Performs piecewise master equation simulation using the Lindblad formalism
    with thermal collapse operators computed at each time segment.

    For Bacon-Shor code, tracks both logical and ancilla (gauge) degrees of freedom
    and returns total fidelity as logical_one + ancilla_one.

    Parameters
    ----------
    config : QECConfigBase
        QEC configuration object (QECConfig or BaconShorConfig)
    initial_state : Qobj
        Initial quantum state (ket)
    target_state : Qobj
        Target state for fidelity calculation (ket)
    Ep : float
        Penalty energy in rad/s
    T_K : float
        Temperature in Kelvin
    lambda_2 : float
        Coupling strength parameter for thermal noise
    n_steps : int, optional
        Number of simulation time steps. If None, uses len(config.t_list)
    return_trajectory : bool, optional
        If True, return additional trajectory info (default False)
    verbose : bool, optional
        If True, print progress (default False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'fidelity': total fidelity (logical_one for rep, logical_one + ancilla_one for BS)
        - 'fidelity_logical': fidelity to logical_one only
        - 'trajectory': fidelity trajectory (if return_trajectory=True)
        - 'trajectory_logical': logical-only trajectory (if return_trajectory=True)

    Example
    -------
    >>> from qec_config import QECConfig
    >>> config = QECConfig(n_points=201)
    >>> Ep = config.Ep_MHz_to_rad(75)
    >>> result = simulate_rap_fidelity(
    ...     config, config.logical_zero, config.logical_one,
    ...     Ep=Ep, T_K=0.02, lambda_2=1e4
    ... )
    >>> print(result['fidelity'])
    """
    # Determine number of qubits and if this is Bacon-Shor
    is_bacon_shor = hasattr(config, 'code_name') and config.code_name == 'bacon_shor'
    n_qubits = 4 if is_bacon_shor else 3

    # Get time list (optionally subsample for speed)
    if n_steps is not None and n_steps < len(config.t_list):
        indices = np.linspace(0, len(config.t_list) - 1, n_steps, dtype=int)
        t_list = config.t_list[indices]
    else:
        t_list = config.t_list

    # Build Hamiltonian function
    H_func = config.make_H_func(Ep=Ep, pulse_type='gaussian')

    # Build expectation operators
    rho_L0 = config.logical_zero * config.logical_zero.dag()
    rho_L1 = config.logical_one * config.logical_one.dag()

    if is_bacon_shor:
        # For Bacon-Shor, also track ancilla (gauge) states
        rho_A0 = config.ancilla_zero * config.ancilla_zero.dag()
        rho_A1 = config.ancilla_one * config.ancilla_one.dag()
        e_ops = [rho_L0, rho_L1, rho_A0, rho_A1]
    else:
        e_ops = [rho_L0, rho_L1]

    # Initialize density matrix
    rho = initial_state * initial_state.dag()

    # Track trajectories
    trajectory_logical = []
    trajectory_total = []

    # Initial fidelities
    fid_L1_init = float((rho_L1 * rho).tr().real)
    trajectory_logical.append(fid_L1_init)
    if is_bacon_shor:
        fid_A1_init = float((rho_A1 * rho).tr().real)
        trajectory_total.append(fid_L1_init + fid_A1_init)
    else:
        trajectory_total.append(fid_L1_init)

    # Solver options (QuTiP 5 uses dict)
    opts = {'store_states': True, 'nsteps': 10000, 'rtol': 1e-6, 'atol': 1e-8}

    # Piecewise simulation
    prev_time = float(t_list[0])

    for k, t_next in enumerate(t_list[1:]):
        t_next = float(t_next)
        t_mid = 0.5 * (prev_time + t_next)

        # Compute thermal collapse operators at segment midpoint
        try:
            c_ops = c_ops_gen_thermal(t_mid, H_func, n_qubits, lambda_2, T_K)
        except Exception:
            c_ops = None

        # Propagate one segment
        t_steps = [prev_time, t_next]
        try:
            result = mesolve(H_func, rho, t_steps, c_ops=c_ops,
                           e_ops=e_ops, options=opts)
        except Exception as e1:
            # Fallback: try without collapse operators
            try:
                result = mesolve(H_func, rho, t_steps, c_ops=None,
                               e_ops=e_ops, options=opts)
            except Exception as exc:
                if verbose:
                    print(f"Solver failed at t={t_mid:.2e}: {exc}")
                # Keep previous values and continue
                trajectory_logical.append(trajectory_logical[-1])
                trajectory_total.append(trajectory_total[-1])
                prev_time = t_next
                continue

        # Update state
        if hasattr(result, 'states') and result.states:
            rho = result.states[-1]
        prev_time = t_next

        # Extract expectation values
        try:
            fid_L1 = float(np.real(result.expect[1][-1]))
            if is_bacon_shor:
                fid_A1 = float(np.real(result.expect[3][-1]))
                fid_total = fid_L1 + fid_A1
            else:
                fid_total = fid_L1
        except Exception:
            fid_L1 = float((rho_L1 * rho).tr().real)
            if is_bacon_shor:
                fid_A1 = float((rho_A1 * rho).tr().real)
                fid_total = fid_L1 + fid_A1
            else:
                fid_total = fid_L1

        trajectory_logical.append(fid_L1)
        trajectory_total.append(fid_total)

        if verbose and (k + 1) % 50 == 0:
            print(f"  Step {k+1}/{len(t_list)-1}, fidelity = {fid_total:.4f}")

    # Compute final fidelities
    final_fid_L1 = float((rho_L1 * rho).tr().real)
    if is_bacon_shor:
        final_fid_A1 = float((rho_A1 * rho).tr().real)
        final_fid_total = final_fid_L1 + final_fid_A1
    else:
        final_fid_total = final_fid_L1

    # Build result dictionary
    result_dict = {
        'fidelity': final_fid_total,
        'fidelity_logical': final_fid_L1,
    }

    if return_trajectory:
        result_dict['trajectory'] = np.array(trajectory_total)
        result_dict['trajectory_logical'] = np.array(trajectory_logical)

    return result_dict


def simulate_fidelity_sweep(config, Ep_values, T_values, lambda_2,
                            n_steps=None, verbose=True):
    """
    Sweep over penalty energies and temperatures to compute fidelity matrix.

    Simulates RAP protocol from |0_L⟩ to |1_L⟩ for each (Ep, T) combination.

    Parameters
    ----------
    config : QECConfigBase
        QEC configuration object
    Ep_values : array-like
        Array of penalty energies in rad/s
    T_values : array-like
        Array of temperatures in Kelvin
    lambda_2 : float
        Coupling strength parameter
    n_steps : int, optional
        Number of simulation time steps (for speed)
    verbose : bool, optional
        If True, print progress

    Returns
    -------
    dict
        Dictionary containing:
        - 'fidelities': 2D array of total fidelities [T_idx, Ep_idx]
        - 'fidelities_logical': 2D array of logical-only fidelities [T_idx, Ep_idx]
        - 'Ep_values': Ep values in rad/s
        - 'Ep_MHz': Ep values in MHz
        - 'T_values': Temperature values in K
    """
    n_Ep = len(Ep_values)
    n_T = len(T_values)

    fidelities = np.zeros((n_T, n_Ep))
    fidelities_logical = np.zeros((n_T, n_Ep))

    initial_state = config.logical_zero
    target_state = config.logical_one

    if verbose:
        print(f"Simulating {config.code_name}: {n_T} temperatures × {n_Ep} Ep values")
        print("=" * 60)

    for i, T in enumerate(T_values):
        if verbose:
            print(f"  T = {T*1e3:.1f} mK ...", end="", flush=True)

        for j, Ep in enumerate(Ep_values):
            result = simulate_rap_fidelity(
                config, initial_state, target_state,
                Ep=Ep, T_K=T, lambda_2=lambda_2,
                n_steps=n_steps, verbose=False
            )
            fidelities[i, j] = result['fidelity']
            fidelities_logical[i, j] = result['fidelity_logical']

        if verbose:
            print(f" done (avg fid = {fidelities[i].mean():.4f})")

    if verbose:
        print("=" * 60)

    return {
        'fidelities': fidelities,
        'fidelities_logical': fidelities_logical,
        'Ep_values': np.array(Ep_values),
        'Ep_MHz': np.array([config.to_MHz(Ep) for Ep in Ep_values]),
        'T_values': np.array(T_values)
    }
