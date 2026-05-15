import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "generic_crystal.yaml"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "output" / "allan_deviation_generic_crystal.png"


def phase_noise_psd_from_piecewise_points(freq_hz, points):
    points_sorted = sorted(points, key=lambda p: p[0])
    p_freq = np.array([p[0] for p in points_sorted], dtype=float)
    p_l_dbc = np.array([p[1] for p in points_sorted], dtype=float)

    if np.any(p_freq <= 0.0):
        raise ValueError("All piecewise phase-noise frequencies must be > 0")
    if np.any(np.diff(p_freq) <= 0.0):
        raise ValueError("Piecewise phase-noise frequencies must be strictly increasing")

    sphi = np.zeros_like(freq_hz, dtype=float)
    mask = freq_hz > 0.0
    if not np.any(mask):
        return sphi

    logf = np.log10(freq_hz[mask])
    logf_points = np.log10(p_freq)
    l_dbc_interp = np.interp(logf, logf_points, p_l_dbc)
    l_lin = 10.0 ** (l_dbc_interp / 10.0)
    sphi[mask] = 2.0 * l_lin
    return sphi


def synthesize_phase_noise_from_psd(freq_bins, sphi, sample_count, df, rng, fmin_hz=0.0, fmax_hz=None):
    if fmax_hz is None:
        fmax_hz = np.max(freq_bins)
    band_mask = (freq_bins >= fmin_hz) & (freq_bins <= fmax_hz)
    sphi = np.where(band_mask, sphi, 0.0)

    spectrum = np.zeros(freq_bins.size, dtype=complex)

    interior = slice(1, -1 if sample_count % 2 == 0 else None)
    bin_sigma = sample_count * np.sqrt(0.5 * sphi[interior] * df)
    gaussian_complex = (rng.normal(size=bin_sigma.size) + 1j * rng.normal(size=bin_sigma.size)) / np.sqrt(2.0)
    spectrum[interior] = gaussian_complex * bin_sigma

    if sample_count % 2 == 0:
        nyq_sigma = sample_count * np.sqrt(sphi[-1] * df)
        spectrum[-1] = rng.normal() * nyq_sigma

    phase_noise_rad = np.fft.irfft(spectrum, n=sample_count)
    phase_noise_rad -= np.mean(phase_noise_rad)
    return phase_noise_rad


def overlapping_allan_deviation_from_time_error(time_error_s, fs_hz, points=50):
    n = time_error_s.size
    tau0 = 1.0 / fs_hz
    max_m = max(1, n // 4)
    m_values = np.unique(np.logspace(0, np.log10(max_m), points).astype(int))

    taus = []
    adev = []
    for m in m_values:
        if n <= 2 * m:
            continue
        d2 = time_error_s[2 * m :] - 2.0 * time_error_s[m:-m] + time_error_s[:-2 * m]
        sigma2 = np.mean(d2 ** 2) / (2.0 * (m * tau0) ** 2)
        taus.append(m * tau0)
        adev.append(np.sqrt(max(sigma2, 0.0)))

    return np.asarray(taus), np.asarray(adev)


def load_demo_settings(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    audio = raw.get("audio", {})
    phase_noise = raw.get("phase_noise", {})
    integration = raw.get("integration", {})

    if str(phase_noise.get("model", "")).strip() != "piecewise":
        raise ValueError("This demo expects phase_noise.model: piecewise")

    points = [(float(p[0]), float(p[1])) for p in phase_noise.get("piecewise_points", [])]
    if len(points) < 2:
        raise ValueError("Need at least two piecewise_points anchors")

    return {
        "fs_audio_hz": float(audio.get("fs_audio_hz", 48_000.0)),
        "duration_s": float(audio.get("duration_s", 20.0)),
        "clock_hz": float(audio.get("clock_hz", 4_915_200.0)),
        "rng_seed": int(audio.get("rng_seed", 42)),
        "piecewise_points": points,
        "fmin_hz": float(integration.get("fmin_hz", 0.1)),
        "fmax_hz": float(integration.get("fmax_hz", 1_000_000.0)),
    }


def main(config_path, output_path, show_plot):
    cfg = load_demo_settings(config_path)

    fs_audio_hz = cfg["fs_audio_hz"]
    duration_s = cfg["duration_s"]
    clock_hz = cfg["clock_hz"]
    rng = np.random.default_rng(cfg["rng_seed"])

    sample_count = int(fs_audio_hz * duration_s)
    freq_bins = np.fft.rfftfreq(sample_count, d=1.0 / fs_audio_hz)
    df = fs_audio_hz / sample_count

    sphi = phase_noise_psd_from_piecewise_points(freq_bins, cfg["piecewise_points"])
    effective_fmax_hz = min(cfg["fmax_hz"], fs_audio_hz / 2.0)
    phase_noise_rad = synthesize_phase_noise_from_psd(
        freq_bins,
        sphi,
        sample_count,
        df,
        rng,
        fmin_hz=cfg["fmin_hz"],
        fmax_hz=effective_fmax_hz,
    )

    jitter_s = phase_noise_rad / (2.0 * np.pi * clock_hz)
    taus_s, adev = overlapping_allan_deviation_from_time_error(jitter_s, fs_audio_hz)

    if taus_s.size == 0:
        raise RuntimeError("Not enough samples to compute Allan deviation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.loglog(taus_s, adev, marker="o", markersize=3, linewidth=1.4)
    ax.set_xlabel("Averaging time tau [s]")
    ax.set_ylabel("Allan deviation sigma_y(tau)")
    ax.set_title("Allan Deviation Demo (generic_crystal profile)")
    ax.grid(True, which="both", alpha=0.35)
    fig.savefig(output_path, dpi=150)

    if show_plot and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Allan deviation demo from generic_crystal phase-noise profile")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Do not show plot window")
    args = parser.parse_args()

    main(args.config, args.output, show_plot=not args.no_show)
