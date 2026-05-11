import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


DEFAULT_CONFIG_PATH = Path(__file__).with_name("clock_audio_jitter_config.yaml")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("clock_audio_jitter_results.png")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML content must be a mapping")

    audio = raw.get("audio", {})
    signal = raw.get("signal", {})
    phase_noise = raw.get("phase_noise", {})
    integration = raw.get("integration", {})
    plots = raw.get("plots", {})

    cfg = {
        "dut_name": raw.get("dut_name", None),
        "output_path": Path(plots.get("output_path", DEFAULT_OUTPUT_PATH)),
        "fs_audio_hz": float(audio.get("fs_audio_hz", 48_000.0)),
        "duration_s": float(audio.get("duration_s", 20.0)),
        "input_tone_hz": float(audio.get("input_tone_hz", 1_000.0)),
        "clock_hz": float(audio.get("clock_hz", 24_576_000.0)),
        "rng_seed": int(audio.get("rng_seed", 42)),
        "waveform_zoom_periods": float(plots.get("waveform_zoom_periods", 5.0)),
        "jitter_overview_fraction": float(plots.get("jitter_overview_fraction", 0.25)),
        "multitone_mode": str(signal.get("mode", "single")),
        "multitone_tones_hz": [float(v) for v in signal.get("multitone_tones_hz", [19_000.0, 20_000.0])],
        "comb_tone_count": int(signal.get("comb_tone_count", 32)),
        "comb_freq_min_hz": float(signal.get("comb_freq_min_hz", 1_000.0)),
        "comb_freq_max_hz": float(signal.get("comb_freq_max_hz", 20_000.0)),
        "phase_noise_model": str(phase_noise.get("model", "power_law")),
        "phase_noise_f1_hz": float(phase_noise.get("power_law", {}).get("f1_hz", 0.1)),
        "phase_noise_l1_dbc": float(phase_noise.get("power_law", {}).get("l1_dbc", -80.0)),
        "phase_noise_f2_hz": float(phase_noise.get("power_law", {}).get("f2_hz", 10_000.0)),
        "phase_noise_l2_dbc": float(phase_noise.get("power_law", {}).get("l2_dbc", -180.0)),
        "use_fixed_slope_model": bool(phase_noise.get("fixed_slope", {}).get("use_legacy_flag", False)),
        "fixed_slope_alpha": float(phase_noise.get("fixed_slope", {}).get("alpha", 1.0)),
        "fixed_slope_ref_freq_hz": float(phase_noise.get("fixed_slope", {}).get("ref_freq_hz", 10_000.0)),
        "fixed_slope_ref_level_dbc": float(phase_noise.get("fixed_slope", {}).get("ref_level_dbc", -180.0)),
        "piecewise_phase_noise_points": [
            (float(p[0]), float(p[1]))
            for p in phase_noise.get("piecewise_points", [(0.1, -80.0), (10_000.0, -180.0)])
        ],
        "jitter_integration_fmin_hz": float(integration.get("fmin_hz", 0.1)),
        "jitter_integration_fmax_hz": float(integration.get("fmax_hz", 24_000.0)),
        "bw_limited_jitter_fmin_hz": float(integration.get("bw_limited_fmin_hz", 100.0)),
        "bw_limited_jitter_fmax_hz": float(
            integration.get("bw_limited_fmax_hz", integration.get("fmax_hz", 24_000.0))
        ),
    }

    if cfg["multitone_mode"] not in {"single", "twotone", "comb"}:
        raise ValueError("signal.mode must be one of: single, twotone, comb")
    if cfg["phase_noise_model"] not in {"power_law", "fixed_slope", "piecewise"}:
        raise ValueError("phase_noise.model must be one of: power_law, fixed_slope, piecewise")
    if cfg["fs_audio_hz"] <= 0.0 or cfg["duration_s"] <= 0.0 or cfg["clock_hz"] <= 0.0:
        raise ValueError("audio.fs_audio_hz, audio.duration_s, and audio.clock_hz must be > 0")
    if cfg["jitter_integration_fmin_hz"] <= 0.0:
        raise ValueError("integration.fmin_hz must be > 0")
    if cfg["bw_limited_jitter_fmin_hz"] <= 0.0:
        raise ValueError("integration.bw_limited_fmin_hz must be > 0")
    if cfg["bw_limited_jitter_fmax_hz"] <= cfg["bw_limited_jitter_fmin_hz"]:
        raise ValueError("integration.bw_limited_fmax_hz must be > integration.bw_limited_fmin_hz")

    dut_name = cfg["dut_name"]
    if dut_name is None:
        cfg["dut_name"] = None
    else:
        dut_name = str(dut_name).strip()
        cfg["dut_name"] = dut_name if dut_name else None

    return cfg


def build_power_law_phase_noise_model(f1_hz, l1_dbc, f2_hz, l2_dbc):
    slope_dbc_per_dec = (l2_dbc - l1_dbc) / (np.log10(f2_hz) - np.log10(f1_hz))
    alpha = -slope_dbc_per_dec / 10.0

    l1_lin = 10.0 ** (l1_dbc / 10.0)
    sphi_1 = 2.0 * l1_lin
    k = sphi_1 * (f1_hz ** alpha)

    return alpha, k


def build_fixed_alpha_phase_noise_model(f_ref_hz, l_ref_dbc, alpha):
    l_ref_lin = 10.0 ** (l_ref_dbc / 10.0)
    sphi_ref = 2.0 * l_ref_lin
    k = sphi_ref * (f_ref_hz ** alpha)
    return alpha, k


def phase_noise_psd_from_model(freq_hz, alpha, k):
    sphi = np.zeros_like(freq_hz, dtype=float)
    mask = freq_hz > 0.0
    sphi[mask] = k / (freq_hz[mask] ** alpha)
    return sphi


def phase_noise_psd_from_piecewise_points(freq_hz, points):
    """Return two-sided phase PSD S_phi(f) from piecewise (f, L_dBc/Hz) anchors."""
    if len(points) < 2:
        raise ValueError("PIECEWISE_PHASE_NOISE_POINTS must contain at least two points")

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


def model_rms_jitter_seconds_from_psd(freqs_hz, sphi, clock_hz):
    l_ssb_lin = 0.5 * sphi
    sigma_phi_sq = np.trapezoid(2.0 * l_ssb_lin, freqs_hz)
    return np.sqrt(np.maximum(sigma_phi_sq, 0.0)) / (2.0 * np.pi * clock_hz)


def model_rms_jitter_seconds(alpha, k, clock_hz, fmin_hz, fmax_hz, points=100_000):
    freqs = np.logspace(np.log10(fmin_hz), np.log10(fmax_hz), points)
    sphi = phase_noise_psd_from_model(freqs, alpha, k)
    return model_rms_jitter_seconds_from_psd(freqs, sphi, clock_hz)


def synthesize_phase_noise(fs_hz, duration_s, alpha, k, rng, fmin_hz=0.0, fmax_hz=None):
    sample_count = int(fs_hz * duration_s)
    freq_bins = np.fft.rfftfreq(sample_count, d=1.0 / fs_hz)
    df = fs_hz / sample_count

    sphi = phase_noise_psd_from_model(freq_bins, alpha, k)
    if fmax_hz is None:
        fmax_hz = fs_hz / 2.0
    return synthesize_phase_noise_from_psd(freq_bins, sphi, sample_count, df, rng, fmin_hz, fmax_hz)


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

    return phase_noise_rad, freq_bins, sphi


def sample_with_clock_jitter(tones_hz, fs_audio_hz, clock_hz, phase_noise_rad):
    """tones_hz: list of tone frequencies in Hz (equal-amplitude sum)."""
    sample_count = phase_noise_rad.size
    nominal_time_s = np.arange(sample_count) / fs_audio_hz
    jitter_s = phase_noise_rad / (2.0 * np.pi * clock_hz)
    jittered_time_s = nominal_time_s + jitter_s

    scale = 1.0 / len(tones_hz)
    sampled = scale * sum(np.sin(2.0 * np.pi * f * jittered_time_s) for f in tones_hz)
    reference = scale * sum(np.sin(2.0 * np.pi * f * nominal_time_s) for f in tones_hz)
    error = sampled - reference

    return nominal_time_s, jitter_s, sampled, reference, error


def one_sided_psd(signal, fs_hz):
    sample_count = signal.size
    window = np.hanning(sample_count)
    signal_w = (signal - np.mean(signal)) * window
    spectrum = np.fft.rfft(signal_w)

    psd = (np.abs(spectrum) ** 2) / (fs_hz * np.sum(window ** 2))
    if sample_count > 2:
        psd[1:-1] *= 2.0

    freq_bins = np.fft.rfftfreq(sample_count, d=1.0 / fs_hz)
    return freq_bins, psd


def fft_db(signal, fs_hz):
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs_hz)
    mag = np.abs(spectrum) / np.sum(window)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-20))
    return freqs[1:], mag_db[1:]


def main(config_path=DEFAULT_CONFIG_PATH, show_plot=True):
    cfg = load_config(config_path)

    dut_name = cfg["dut_name"]
    output_path = cfg["output_path"]
    fs_audio_hz = cfg["fs_audio_hz"]
    duration_s = cfg["duration_s"]
    input_tone_hz = cfg["input_tone_hz"]
    clock_hz = cfg["clock_hz"]
    rng_seed = cfg["rng_seed"]
    waveform_zoom_periods = cfg["waveform_zoom_periods"]
    jitter_overview_fraction = cfg["jitter_overview_fraction"]

    multitone_mode = cfg["multitone_mode"]
    multitone_tones_hz = cfg["multitone_tones_hz"]
    comb_tone_count = cfg["comb_tone_count"]
    comb_freq_min_hz = cfg["comb_freq_min_hz"]
    comb_freq_max_hz = cfg["comb_freq_max_hz"]

    phase_noise_model = cfg["phase_noise_model"]
    phase_noise_f1_hz = cfg["phase_noise_f1_hz"]
    phase_noise_l1_dbc = cfg["phase_noise_l1_dbc"]
    phase_noise_f2_hz = cfg["phase_noise_f2_hz"]
    phase_noise_l2_dbc = cfg["phase_noise_l2_dbc"]
    use_fixed_slope_model = cfg["use_fixed_slope_model"]
    fixed_slope_alpha = cfg["fixed_slope_alpha"]
    fixed_slope_ref_freq_hz = cfg["fixed_slope_ref_freq_hz"]
    fixed_slope_ref_level_dbc = cfg["fixed_slope_ref_level_dbc"]
    piecewise_phase_noise_points = cfg["piecewise_phase_noise_points"]
    jitter_integration_fmin_hz = cfg["jitter_integration_fmin_hz"]
    jitter_integration_fmax_hz = cfg["jitter_integration_fmax_hz"]
    bw_limited_jitter_fmin_hz = cfg["bw_limited_jitter_fmin_hz"]
    bw_limited_jitter_fmax_hz = cfg["bw_limited_jitter_fmax_hz"]

    rng = np.random.default_rng(rng_seed)
    effective_fmax_hz = min(jitter_integration_fmax_hz, fs_audio_hz / 2.0)

    sample_count = int(fs_audio_hz * duration_s)
    model_freq_bins = np.fft.rfftfreq(sample_count, d=1.0 / fs_audio_hz)
    model_label = ""
    alpha = np.nan

    if phase_noise_model == "piecewise":
        model_sphi = phase_noise_psd_from_piecewise_points(model_freq_bins, piecewise_phase_noise_points)
        model_label = "Piecewise-linear model"
    else:
        # Backward compatibility: legacy boolean still works when model is not explicitly piecewise.
        use_fixed = (phase_noise_model == "fixed_slope") or (
            phase_noise_model == "power_law" and use_fixed_slope_model
        )
        if use_fixed:
            alpha, k = build_fixed_alpha_phase_noise_model(
                fixed_slope_ref_freq_hz,
                fixed_slope_ref_level_dbc,
                fixed_slope_alpha,
            )
            model_label = f"Target 1/f^{alpha:.1f} model (fixed slope)"
        else:
            alpha, k = build_power_law_phase_noise_model(
                phase_noise_f1_hz,
                phase_noise_l1_dbc,
                phase_noise_f2_hz,
                phase_noise_l2_dbc,
            )
            model_label = f"Target 1/f^{alpha:.1f} model"
        model_sphi = phase_noise_psd_from_model(model_freq_bins, alpha, k)

    df = fs_audio_hz / sample_count
    phase_noise_rad, _, model_sphi = synthesize_phase_noise_from_psd(
        model_freq_bins,
        model_sphi,
        sample_count,
        df,
        rng,
        fmin_hz=jitter_integration_fmin_hz,
        fmax_hz=effective_fmax_hz,
    )

    jitter_model_mask = (model_freq_bins >= jitter_integration_fmin_hz) & (model_freq_bins <= effective_fmax_hz)
    model_jitter_rms_s = model_rms_jitter_seconds_from_psd(
        model_freq_bins[jitter_model_mask],
        model_sphi[jitter_model_mask],
        clock_hz,
    )
    bw_limited_effective_fmax_hz = min(bw_limited_jitter_fmax_hz, fs_audio_hz / 2.0)
    bw_limited_jitter_mask = (
        (model_freq_bins >= bw_limited_jitter_fmin_hz)
        & (model_freq_bins <= bw_limited_effective_fmax_hz)
    )
    if np.any(bw_limited_jitter_mask):
        model_jitter_bw_limited_rms_s = model_rms_jitter_seconds_from_psd(
            model_freq_bins[bw_limited_jitter_mask],
            model_sphi[bw_limited_jitter_mask],
            clock_hz,
        )
    else:
        model_jitter_bw_limited_rms_s = np.nan

    # Resolve active tone list from mode
    if multitone_mode == "comb":
        active_tones = list(np.linspace(comb_freq_min_hz, comb_freq_max_hz, comb_tone_count))
    elif multitone_mode == "twotone":
        active_tones = list(multitone_tones_hz)
    else:
        active_tones = [input_tone_hz]

    time_s, jitter_s, sampled, reference, error = sample_with_clock_jitter(
        active_tones,
        fs_audio_hz,
        clock_hz,
        phase_noise_rad,
    )

    jitter_rms_s = np.sqrt(np.mean(jitter_s ** 2))
    error_rms = np.sqrt(np.mean(error ** 2))
    ref_rms = np.sqrt(np.mean(reference ** 2))
    snr_jitter_db = 20.0 * np.log10(ref_rms / np.maximum(error_rms, 1e-20))

    psd_freq, psd_phase = one_sided_psd(phase_noise_rad, fs_audio_hz)
    phase_noise_est_dbc = 10.0 * np.log10(np.maximum(psd_phase / 2.0, 1e-30))

    model_phase_noise_dbc = 10.0 * np.log10(np.maximum(model_sphi / 2.0, 1e-30))

    fft_freq_ref, fft_db_ref = fft_db(reference, fs_audio_hz)
    fft_freq_smp, fft_db_smp = fft_db(sampled, fs_audio_hz)

    if multitone_mode != "single":
        fft_band_low = 0.0
        fft_band_high = fs_audio_hz / 2.0
        near_tone = np.ones(fft_freq_ref.size, dtype=bool)
        # Zoom: beat period for 2-tone, spacing period for comb, else lowest tone
        if len(active_tones) >= 2:
            zoom_ref_hz = abs(active_tones[1] - active_tones[0])
        else:
            zoom_ref_hz = active_tones[0]
    else:
        fft_band_low = input_tone_hz * 0.99
        fft_band_high = input_tone_hz * 1.01
        near_tone = (fft_freq_ref >= fft_band_low) & (fft_freq_ref <= fft_band_high)
        zoom_ref_hz = input_tone_hz

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    slope_label = model_label
    plot_mask_model = model_freq_bins >= jitter_integration_fmin_hz
    plot_mask_est = psd_freq >= jitter_integration_fmin_hz
    axes[0, 0].plot(
        psd_freq[plot_mask_est],
        phase_noise_est_dbc[plot_mask_est],
        label="Synthesized phase noise",
        color="tab:orange",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    axes[0, 0].plot(
        model_freq_bins[plot_mask_model],
        model_phase_noise_dbc[plot_mask_model],
        label=slope_label,
        color="tab:blue",
        linewidth=2.2,
        zorder=8,
    )
    if bw_limited_effective_fmax_hz > bw_limited_jitter_fmin_hz:
        axes[0, 0].axvspan(
            bw_limited_jitter_fmin_hz,
            bw_limited_effective_fmax_hz,
            color="#A6E3BC",
            alpha=0.20,
            zorder=1,
            label=f"BW-limited jitter band ({bw_limited_jitter_fmin_hz:g}..{bw_limited_effective_fmax_hz:g} Hz)",
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Offset frequency [Hz]")
    axes[0, 0].set_ylabel("L(f) [dBc/Hz]")
    axes[0, 0].set_title("Clock phase noise profile")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend()

    zoom_count = int(min(np.ceil(waveform_zoom_periods * fs_audio_hz / zoom_ref_hz), time_s.size))
    overview_fraction = np.clip(jitter_overview_fraction, 1.0 / max(time_s.size, 1), 1.0)
    overview_count = int(max(1, np.ceil(overview_fraction * time_s.size)))
    overview_duration_s = overview_count / fs_audio_hz
    axes[0, 1].plot(time_s[:zoom_count], jitter_s[:zoom_count] * 1e12)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Timing error [ps]")
    axes[0, 1].set_title("Clock jitter (time domain, zoom)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(fft_freq_smp[near_tone], fft_db_smp[near_tone], label="Jittered sampling")
    axes[1, 0].plot(
        fft_freq_ref[near_tone],
        fft_db_ref[near_tone],
        label="Ideal sampling",
        color="black",
        linewidth=2.0,
        zorder=10,
    )
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Magnitude [dB]")
    if multitone_mode == "comb":
        axes[1, 0].set_title(
            f"FFT — {comb_tone_count}-tone comb "
            f"{comb_freq_min_hz/1e3:.3g}–{comb_freq_max_hz/1e3:.3g} kHz"
        )
    elif multitone_mode == "twotone":
        tones_str = " + ".join(f"{f/1e3:.3g} kHz" for f in active_tones)
        axes[1, 0].set_title(f"FFT — 2-tone {tones_str}")
    else:
        axes[1, 0].set_title(f"FFT around {input_tone_hz:.0f} Hz tone")
    axes[1, 0].set_xlim(fft_band_low, fft_band_high)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Annotate IMD2 product for 2-tone test
    if multitone_mode == "twotone" and len(active_tones) == 2:
        f_imd2 = abs(active_tones[1] - active_tones[0])
        imd2_idx = np.argmin(np.abs(fft_freq_smp - f_imd2))
        imd2_level_db = fft_db_smp[imd2_idx]
        sig_level_db = max(fft_db_smp[np.argmin(np.abs(fft_freq_smp - f))] for f in active_tones)
        imd2_ratio_db = imd2_level_db - sig_level_db
        axes[1, 0].axvline(f_imd2, color="red", linestyle="--", linewidth=0.8,
                           label=f"IMD2 @ {f_imd2:.0f} Hz ({imd2_ratio_db:+.1f} dBc)")
        axes[1, 0].legend()

    axes[1, 1].plot(time_s[:overview_count], jitter_s[:overview_count] * 1e12)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Timing error [ps]")
    axes[1, 1].set_title(
        f"Clock jitter (time domain, {overview_duration_s:g} s view, {overview_fraction * 100.0:.0f}% of run)"
    )
    axes[1, 1].grid(True, alpha=0.3)

    if multitone_mode == "comb":
        mode_label = f"{comb_tone_count}-tone comb {comb_freq_min_hz/1e3:.3g}–{comb_freq_max_hz/1e3:.3g} kHz"
    elif multitone_mode == "twotone":
        mode_label = "2-tone " + " + ".join(f"{f/1e3:.3g}kHz" for f in active_tones)
    else:
        mode_label = f"{input_tone_hz:.0f} Hz single tone"
    profile_label = f"1/f^{alpha:.1f}" if np.isfinite(alpha) else "piecewise"
    heading_prefix = f"DUT: {dut_name} — " if dut_name else ""
    fig.suptitle(
        f"{heading_prefix}{mode_label}\n"
        f"RMS jitter(sim) = {jitter_rms_s * 1e12:.3f} ps, "
        f"RMS jitter(model {jitter_integration_fmin_hz:g}..{effective_fmax_hz:g} Hz) = {model_jitter_rms_s * 1e12:.3f} ps, "
        f"RMS jitter(bw-limited {bw_limited_jitter_fmin_hz:g}..{bw_limited_effective_fmax_hz:g} Hz) = {model_jitter_bw_limited_rms_s * 1e12:.3f} ps, "
        f"jitter-limited SNR = {snr_jitter_db:.2f} dB"
    )

    fig.savefig(output_path, dpi=150)

    if show_plot and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clock phase-noise to audio jitter impact simulator")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH.name})",
    )
    args = parser.parse_args()
    main(config_path=args.config)
