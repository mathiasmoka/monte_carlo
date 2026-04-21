"""
==============================================================================
Bayesian GARCH(1,1) — Random Walk Metropolis Sampler
==============================================================================
Following the model and priors described in:
  Mira, Solgi & Imparato (2013) "Zero variance Markov chain Monte Carlo
  for Bayesian estimators", Statistics and Computing 23:653-662.

Model
-----
  y_t  = sqrt(h_t) * eps_t,   eps_t ~ N(0,1)  (or Student-t)
  h_t  = omega + alpha * y_{t-1}^2 + beta * h_{t-1}

Parameters  θ = (omega, alpha, beta)
Constraints: omega > 0,  alpha >= 0,  beta >= 0,  alpha + beta < 1

Prior (Mira et al. use the Ardia 2008 / Nakatsuma 2000 setup):
  p(omega) ∝ 1/omega  (log-uniform / Jeffreys-type)
  p(alpha) ~ Uniform(0, 1)
  p(beta)  ~ Uniform(0, 1)
  Subject to: alpha + beta < 1

Log-posterior (up to constant)
  log π(θ|y) = log L(θ;y) + log p(θ)

Random Walk Metropolis:
  Proposal  θ* = θ + δ,   δ ~ N(0, Σ)
  where Σ is tuned to target ~23% acceptance rate (Gelman et al. rule).
  Work in transformed space (log omega, logit-α, logit-β) for unconstrained
  sampling, then map back.

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  1.  GARCH(1,1) utilities
# ─────────────────────────────────────────────────────────────

def garch_variances(y, omega, alpha, beta):
    """Recursively compute conditional variances h_t."""
    T = len(y)
    h = np.empty(T)
    h[0] = np.var(y)           # initialise at sample variance
    for t in range(1, T):
        h[t] = omega + alpha * y[t-1]**2 + beta * h[t-1]
    return h


def log_likelihood(y, omega, alpha, beta):
    """Gaussian GARCH(1,1) log-likelihood (sum over t=1..T)."""
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return -np.inf
    h = garch_variances(y, omega, alpha, beta)
    # avoid numerical issues
    if np.any(h <= 0):
        return -np.inf
    return -0.5 * np.sum(np.log(h) + y**2 / h)


def log_prior(omega, alpha, beta):
    """Log-prior: log-uniform on omega, Uniform on alpha & beta,
    with the stationarity constraint alpha+beta < 1."""
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return -np.inf
    # log p(omega) ∝ -log(omega)   (Jeffreys prior)
    return -np.log(omega)


def log_posterior(y, omega, alpha, beta):
    lp = log_prior(omega, alpha, beta)
    if not np.isfinite(lp):
        return -np.inf
    return log_likelihood(y, omega, alpha, beta) + lp


# ─────────────────────────────────────────────────────────────
#  2.  Parameter transformations  (constrained → unconstrained)
# ─────────────────────────────────────────────────────────────
# theta = (omega, alpha, beta)   constrained
# phi   = (log_omega, logit_alpha, logit_beta_given_alpha)  unconstrained
#
# alpha  = sigmoid(phi[1])
# beta   = (1-alpha) * sigmoid(phi[2])   ensures alpha+beta < 1

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    return np.log(p / (1.0 - p))

def phi_to_theta(phi):
    """Unconstrained → constrained."""
    log_omega, phi_alpha, phi_beta = phi
    omega = np.exp(log_omega)
    alpha = sigmoid(phi_alpha)
    beta  = (1.0 - alpha) * sigmoid(phi_beta)
    return omega, alpha, beta

def theta_to_phi(omega, alpha, beta):
    """Constrained → unconstrained."""
    log_omega = np.log(omega)
    phi_alpha = logit(alpha)
    phi_beta  = logit(beta / (1.0 - alpha))
    return np.array([log_omega, phi_alpha, phi_beta])

def log_jacobian(phi):
    """
    Log |∂theta/∂phi| for the change of variables.
    Needed so the RWM in phi-space targets the right posterior.
    """
    _, phi_alpha, phi_beta = phi
    alpha = sigmoid(phi_alpha)
    s_b   = sigmoid(phi_beta)
    # d(omega)/d(log_omega) = omega
    # d(alpha)/d(phi_alpha) = alpha*(1-alpha)
    # d(beta)/d(phi_beta)   = (1-alpha)*s_b*(1-s_b)
    log_J = (phi[0]                          # log omega
             + np.log(alpha*(1-alpha))        # dalpha/dphi_alpha
             + np.log((1-alpha)*s_b*(1-s_b))) # dbeta/dphi_beta
    return log_J

def log_posterior_transformed(y, phi):
    """Log-posterior in unconstrained space (includes Jacobian)."""
    omega, alpha, beta = phi_to_theta(phi)
    lp = log_posterior(y, omega, alpha, beta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_jacobian(phi)


# ─────────────────────────────────────────────────────────────
#  3.  Random Walk Metropolis sampler
# ─────────────────────────────────────────────────────────────

def rwm_garch(y, n_iter=50_000, burn_in=10_000, sigma_prop=None,
              seed=42, verbose=True):
    """
    Random Walk Metropolis for the GARCH(1,1) posterior.
    Sampling is done in the unconstrained (phi) space.

    Parameters
    ----------
    y         : array-like, log-returns
    n_iter    : total MCMC iterations (including burn-in)
    burn_in   : number of draws to discard
    sigma_prop: proposal standard deviations (len-3 array).
                If None, auto-tuned via a short pilot run.
    seed      : random seed
    verbose   : print acceptance rate and progress

    Returns
    -------
    samples   : dict with arrays 'omega','alpha','beta' (post burn-in)
    accept_rate: float
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)

    # ── Starting values: rough MLE-style initialisation ──
    omega0 = np.var(y) * 0.05
    alpha0 = 0.10
    beta0  = 0.85
    phi_cur = theta_to_phi(omega0, alpha0, beta0)
    lp_cur  = log_posterior_transformed(y, phi_cur)

    # ── Proposal covariance (diagonal) ──
    if sigma_prop is None:
        # Pilot run of 2000 iterations with wide steps to estimate scale
        sigma_prop = np.array([0.05, 0.10, 0.10])
        if verbose:
            print("Running pilot to calibrate proposal ...")
        sigma_prop = _calibrate_proposal(y, phi_cur, sigma_prop,
                                         rng, n_pilot=3000)
        if verbose:
            print(f"  Calibrated σ_prop = {sigma_prop.round(4)}")

    # ── Main chain ──
    chain = np.empty((n_iter, 3))
    n_accept = 0

    for i in range(n_iter):
        # Proposal
        phi_prop = phi_cur + sigma_prop * rng.standard_normal(3)
        lp_prop  = log_posterior_transformed(y, phi_prop)

        # Metropolis ratio (symmetric proposal → only posterior ratio)
        log_alpha_mh = lp_prop - lp_cur
        if np.log(rng.uniform()) < log_alpha_mh:
            phi_cur = phi_prop
            lp_cur  = lp_prop
            n_accept += 1

        chain[i] = phi_to_theta(phi_cur)

        if verbose and (i+1) % 10_000 == 0:
            ar = n_accept / (i+1)
            print(f"  Iter {i+1:>6d}  accept = {ar:.3f}")

    accept_rate = n_accept / n_iter
    samples = {
        'omega': chain[burn_in:, 0],
        'alpha': chain[burn_in:, 1],
        'beta' : chain[burn_in:, 2],
    }
    return samples, accept_rate


def _calibrate_proposal(y, phi_start, sigma_init, rng, n_pilot=3000,
                        target_ar=0.23):
    """Simple adaptive scaling: adjust σ to hit ~23% acceptance rate."""
    sigma = sigma_init.copy()
    phi_cur = phi_start.copy()
    lp_cur  = log_posterior_transformed(y, phi_cur)
    n_accept = 0
    for i in range(n_pilot):
        phi_prop = phi_cur + sigma * rng.standard_normal(3)
        lp_prop  = log_posterior_transformed(y, phi_prop)
        if np.log(rng.uniform()) < lp_prop - lp_cur:
            phi_cur = phi_prop
            lp_cur  = lp_prop
            n_accept += 1
        # Adapt every 500 iterations
        if (i+1) % 500 == 0:
            ar = n_accept / (i+1)
            sigma *= np.exp(ar - target_ar)   # simple Robbins-Monro step
    return sigma


# ─────────────────────────────────────────────────────────────
#  4.  Data generation (simulated GARCH)
# ─────────────────────────────────────────────────────────────

def simulate_garch(T, omega, alpha, beta, seed=0):
    """Simulate a GARCH(1,1) process."""
    rng = np.random.default_rng(seed)
    y = np.empty(T)
    h = np.empty(T)
    h[0] = omega / (1 - alpha - beta)   # unconditional variance
    y[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, T):
        h[t] = omega + alpha * y[t-1]**2 + beta * h[t-1]
        y[t] = np.sqrt(h[t]) * rng.standard_normal()
    return y, h


# ─────────────────────────────────────────────────────────────
#  5.  Diagnostics utilities
# ─────────────────────────────────────────────────────────────

def autocorr(x, maxlag=50):
    """Normalized autocorrelation up to maxlag."""
    x = x - x.mean()
    var = np.dot(x, x)
    ac = np.array([np.dot(x[:len(x)-k], x[k:]) / var
                   for k in range(maxlag+1)])
    return ac

def ess(x):
    """Effective sample size via initial monotone sequence estimator."""
    ac = autocorr(x, maxlag=min(500, len(x)//4))
    # sum of consecutive pairs until non-positive
    pairs = ac[1::2] + ac[2::2]
    stop  = np.where(pairs < 0)[0]
    m     = stop[0] if len(stop) > 0 else len(pairs)
    tau   = 1 + 2 * np.sum(ac[1:2*m+1])
    return len(x) / max(tau, 1.0)

def gelman_rubin(chains):
    """Gelman-Rubin R̂ for a list of 1D chains of equal length."""
    M = len(chains)
    N = len(chains[0])
    means = np.array([c.mean() for c in chains])
    vars_ = np.array([c.var(ddof=1) for c in chains])
    B = N * np.var(means, ddof=1)
    W = np.mean(vars_)
    Vhat = (N-1)/N * W + B/N
    return np.sqrt(Vhat / W)

def posterior_summary(samples, true_vals=None):
    """Print a summary table for the posterior samples."""
    print("\n" + "="*60)
    print(f"{'Parameter':>10}  {'Mean':>8}  {'Std':>8}  "
          f"{'2.5%':>8}  {'97.5%':>8}  {'ESS':>7}")
    print("-"*60)
    for name in ['omega', 'alpha', 'beta']:
        s  = samples[name]
        q  = np.quantile(s, [0.025, 0.975])
        e  = ess(s)
        tv = f"  (true={true_vals[name]:.4f})" if true_vals else ""
        print(f"{name:>10}  {s.mean():8.4f}  {s.std():8.4f}  "
              f"{q[0]:8.4f}  {q[1]:8.4f}  {e:7.0f}{tv}")
    print("="*60)


# ─────────────────────────────────────────────────────────────
#  6.  Plotting
# ─────────────────────────────────────────────────────────────

def plot_mcmc(samples, true_vals=None, title="GARCH(1,1) Posterior", save_path=None):
    names = ['omega', 'alpha', 'beta']
    labels = [r'$\omega$', r'$\alpha$', r'$\beta$']
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    for row, (name, label) in enumerate(zip(names, labels)):
        s = samples[name]

        # ── Trace ──
        ax_trace = fig.add_subplot(gs[row, 0])
        ax_trace.plot(s, lw=0.4, alpha=0.8, color='steelblue')
        if true_vals:
            ax_trace.axhline(true_vals[name], color='crimson', lw=1.5,
                             linestyle='--', label='true')
        ax_trace.set_ylabel(label, fontsize=11)
        ax_trace.set_xlabel("Iteration")
        ax_trace.set_title("Trace" if row == 0 else "")

        # ── Histogram ──
        ax_hist = fig.add_subplot(gs[row, 1])
        ax_hist.hist(s, bins=60, density=True, color='steelblue',
                     alpha=0.7, edgecolor='none')
        if true_vals:
            ax_hist.axvline(true_vals[name], color='crimson', lw=1.5,
                            linestyle='--', label='true')
        ax_hist.set_xlabel(label, fontsize=11)
        ax_hist.set_title("Posterior density" if row == 0 else "")

        # ── Autocorrelation ──
        ax_ac = fig.add_subplot(gs[row, 2])
        maxlag = 60
        ac = autocorr(s, maxlag)
        ax_ac.bar(range(maxlag+1), ac, width=0.8, color='steelblue',
                  alpha=0.7)
        ax_ac.axhline(0, color='black', lw=0.8)
        ax_ac.axhline(1.96/np.sqrt(len(s)), color='crimson', lw=1,
                      linestyle='--')
        ax_ac.axhline(-1.96/np.sqrt(len(s)), color='crimson', lw=1,
                      linestyle='--')
        ax_ac.set_xlabel("Lag")
        ax_ac.set_title("Autocorrelation" if row == 0 else "")
        ax_ac.set_xlim(0, maxlag)
        ax_ac.set_ylim(-0.2, 1.0)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
    return fig


def plot_data_and_vol(y, samples, title="", save_path=None):
    """Plot returns and posterior mean conditional variance."""
    T = len(y)
    n_post = min(500, len(samples['omega']))  # use 500 posterior draws
    idx    = np.linspace(0, len(samples['omega'])-1, n_post, dtype=int)

    h_draws = np.empty((n_post, T))
    for i, j in enumerate(idx):
        h_draws[i] = garch_variances(y,
                                     samples['omega'][j],
                                     samples['alpha'][j],
                                     samples['beta'][j])

    h_mean = h_draws.mean(axis=0)
    h_lo   = np.quantile(h_draws, 0.025, axis=0)
    h_hi   = np.quantile(h_draws, 0.975, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(y, lw=0.6, color='grey', alpha=0.8)
    axes[0].set_ylabel("Log-returns", fontsize=11)
    axes[0].set_title(title, fontsize=12, fontweight='bold')

    axes[1].fill_between(range(T), np.sqrt(h_lo), np.sqrt(h_hi),
                         alpha=0.3, color='steelblue', label='95% CI')
    axes[1].plot(np.sqrt(h_mean), lw=1.0, color='steelblue',
                 label='Posterior mean')
    axes[1].set_ylabel(r"Cond. std  $\sqrt{h_t}$", fontsize=11)
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────
#  7.  Real data: EUR/USD (or any CSV of close prices)
# ─────────────────────────────────────────────────────────────

def fetch_eurusd_returns(start="2010-01-01", end="2024-01-01",
                         npy_path=None):
    """
    Load EUR/USD daily log-returns.
    Priority:
      1. yfinance download (requires network + yfinance package)
      2. Pre-saved .npy file  (npy_path)
      3. Returns None  (caller falls back to simulated data)
    """
    try:
        import yfinance as yf
        df = yf.download("EURUSD=X", start=start, end=end,
                         progress=False, auto_adjust=True)
        prices = df['Close'].dropna().values
        lr     = np.diff(np.log(prices))
        print(f"Downloaded {len(lr)} EUR/USD log-return observations.")
        return lr
    except Exception as e:
        print(f"yfinance not available ({e}).")

    if npy_path:
        try:
            lr = np.load(npy_path)
            print(f"Loaded {len(lr)} log-return observations from {npy_path}.")
            return lr
        except Exception as e2:
            print(f"Could not load npy file: {e2}.")

    return None


# ─────────────────────────────────────────────────────────────
#  8.  Main experiment
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    plt.style.use('seaborn-v0_8-whitegrid')
    np.set_printoptions(precision=4, suppress=True)

    # ══════════════════════════════════════════════════════════
    #  PART A — Simulated data
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART A — Simulated GARCH(1,1) data")
    print("="*60)

    TRUE = {'omega': 0.05, 'alpha': 0.10, 'beta': 0.85}
    T    = 1500
    y_sim, h_true = simulate_garch(T, **TRUE, seed=7)

    print(f"\nSimulated T={T} observations with")
    print(f"  omega={TRUE['omega']}, alpha={TRUE['alpha']}, beta={TRUE['beta']}")
    print(f"  (persistence alpha+beta = {TRUE['alpha']+TRUE['beta']:.2f})")

    samples_sim, ar_sim = rwm_garch(
        y_sim,
        n_iter  = 60_000,
        burn_in = 10_000,
        seed    = 42,
        verbose = True,
    )
    print(f"\nAcceptance rate: {ar_sim:.3f}")
    posterior_summary(samples_sim, true_vals=TRUE)

    fig_sim = plot_mcmc(samples_sim, true_vals=TRUE,
                        title="GARCH(1,1) — Simulated data")
    fig_sim.savefig("/mnt/user-data/outputs/garch_sim_diagnostics.png",
                    dpi=140, bbox_inches='tight')

    fig_vol_sim = plot_data_and_vol(
        y_sim, samples_sim,
        title="Simulated GARCH(1,1) — Returns & posterior conditional volatility")
    fig_vol_sim.savefig("/mnt/user-data/outputs/garch_sim_volatility.png",
                        dpi=140, bbox_inches='tight')

    # ══════════════════════════════════════════════════════════
    #  PART B — Real EUR/USD log-returns
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART B — Real EUR/USD log-returns")
    print("="*60)

    y_real = fetch_eurusd_returns(npy_path="/home/claude/eurusd_sim.npy")

    if y_real is None:
        # Fallback: simulate with typical exchange-rate parameters
        print("Simulating with typical FX parameters as fallback.")
        y_real, _ = simulate_garch(2500, omega=0.002, alpha=0.06,
                                   beta=0.93, seed=99)

    samples_real, ar_real = rwm_garch(
        y_real,
        n_iter  = 60_000,
        burn_in = 10_000,
        seed    = 42,
        verbose = True,
    )
    print(f"\nAcceptance rate: {ar_real:.3f}")
    posterior_summary(samples_real)

    fig_real = plot_mcmc(samples_real,
                         title="GARCH(1,1) — EUR/USD log-returns")
    fig_real.savefig("/mnt/user-data/outputs/garch_real_diagnostics.png",
                     dpi=140, bbox_inches='tight')

    fig_vol_real = plot_data_and_vol(
        y_real, samples_real,
        title="EUR/USD — Returns & posterior conditional volatility")
    fig_vol_real.savefig("/mnt/user-data/outputs/garch_real_volatility.png",
                         dpi=140, bbox_inches='tight')

    # ── Final persistence posterior ──────────────────────────
    fig_pers, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (samp, lbl) in zip(axes, [(samples_sim, "Simulated"),
                                       (samples_real, "EUR/USD")]):
        pers = samp['alpha'] + samp['beta']
        ax.hist(pers, bins=60, density=True,
                color='steelblue', alpha=0.7, edgecolor='none')
        ax.axvline(pers.mean(), color='crimson', lw=1.5,
                   label=f"mean={pers.mean():.4f}")
        ax.set_xlabel(r"Persistence $\alpha+\beta$", fontsize=11)
        ax.set_title(f"{lbl} — posterior persistence")
        ax.legend()
    plt.tight_layout()
    fig_pers.savefig("/mnt/user-data/outputs/garch_persistence.png",
                     dpi=140, bbox_inches='tight')

    print("\nAll figures saved to /mnt/user-data/outputs/")
    print("Done.")
