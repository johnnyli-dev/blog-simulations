"""
Generates a two-panel figure comparing:
  (left)  Adaptive step size h(t) from RK45 on the Van der Pol oscillator (mu=5).
  (right) Jacobson/Karels RTO tracking a noisy RTT stream with queueing spikes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def van_der_pol_step_sizes(mu=5.0, t_end=40.0, rtol=1e-6, atol=1e-9):
    def f(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    sol = solve_ivp(
        f,
        (0.0, t_end),
        [2.0, 0.0],
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    t = sol.t
    h = np.diff(t)
    t_mid = 0.5 * (t[:-1] + t[1:])
    return t, sol.y, t_mid, h


def jacobson_karels(samples, alpha=1.0 / 8, beta=1.0 / 4, k=4, srtt0=None, rttvar0=None):
    """Standard RFC 6298 / Jacobson-Karels recursion.

    On the first sample:
        SRTT = R
        RTTVAR = R / 2
    On subsequent samples:
        RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - R|
        SRTT   = (1 - alpha) * SRTT + alpha * R
        RTO    = SRTT + k * RTTVAR
    """
    srtt_arr = np.empty_like(samples, dtype=float)
    rttvar_arr = np.empty_like(samples, dtype=float)
    rto_arr = np.empty_like(samples, dtype=float)

    srtt = samples[0] if srtt0 is None else srtt0
    rttvar = samples[0] / 2.0 if rttvar0 is None else rttvar0
    rto = srtt + k * rttvar
    srtt_arr[0], rttvar_arr[0], rto_arr[0] = srtt, rttvar, rto

    for i in range(1, len(samples)):
        r = samples[i]
        rttvar = (1 - beta) * rttvar + beta * abs(srtt - r)
        srtt = (1 - alpha) * srtt + alpha * r
        rto = srtt + k * rttvar
        srtt_arr[i] = srtt
        rttvar_arr[i] = rttvar
        rto_arr[i] = rto

    return srtt_arr, rttvar_arr, rto_arr


def simulate_rtt_stream(n=400, seed=2):
    """Lognormal RTT around a 50 ms baseline with three queueing bursts up to ~200 ms."""
    rng = np.random.default_rng(seed)
    baseline_ms = 50.0
    sigma = 0.15
    base_samples = rng.lognormal(mean=np.log(baseline_ms), sigma=sigma, size=n)

    bursts = [(80, 100), (200, 215), (310, 335)]
    burst_target = 200.0
    samples = base_samples.copy()
    for start, end in bursts:
        for i in range(start, end):
            decay = 1.0 - (i - start) / max(end - start, 1)
            extra = (burst_target - baseline_ms) * (0.6 + 0.4 * decay)
            samples[i] = base_samples[i] + extra + rng.normal(0, 8)
    return samples


def main():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

    t, y, t_mid, h = van_der_pol_step_sizes(mu=5.0, t_end=40.0)

    ax_l_bg = ax_l.twinx()
    ax_l_bg.plot(t, y[0], color="0.85", linewidth=0.8, zorder=1)
    ax_l_bg.set_yticks([])
    ax_l_bg.set_ylabel("")

    ax_l.plot(t_mid, h, color="#1f4e79", linewidth=1.2, zorder=3)
    ax_l.set_yscale("log")
    ax_l.set_xlabel("t")
    ax_l.set_ylabel("step size h")
    ax_l.set_title(r"RK45 on Van der Pol ($\mu=5$)")
    ax_l.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax_l.set_xlim(0, 40)

    samples = simulate_rtt_stream(n=400, seed=2)
    srtt, rttvar, rto = jacobson_karels(samples)
    x = np.arange(len(samples))

    ax_r.scatter(x, samples, s=8, color="#888888", alpha=0.7, label="RTT sample", zorder=2)
    ax_r.plot(x, srtt, color="#2a7a2a", linewidth=1.0, label="SRTT", zorder=3)
    ax_r.plot(x, rto, color="#b22222", linewidth=1.4, label="RTO", zorder=4)
    ax_r.set_xlabel("sample number")
    ax_r.set_ylabel("milliseconds")
    ax_r.set_title("Jacobson/Karels RTO on a noisy RTT stream")
    ax_r.grid(True, linewidth=0.3, alpha=0.4)
    ax_r.legend(loc="upper right", frameon=False, fontsize=9)
    ax_r.set_xlim(0, len(samples))

    fig.tight_layout()
    out_path = "figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
