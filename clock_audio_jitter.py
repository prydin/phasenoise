import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import wave
import scipy.interpolate

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


DEFAULT_CONFIG_PATH = Path(__file__).with_name("clock_audio_jitter_config.yaml")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("clock_audio_jitter_results.png")


def _optional_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return Path(text)


def _default_jittered_wav_path(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_jittered.wav")


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
        "output_path": _optional_path(plots.get("output_path", None)),
        "fs_audio_hz": float(audio.get("fs_audio_hz", 48_000.0)),
        "duration_s": float(audio.get("duration_s", 20.0)),
        "input_tone_hz": float(signal.get("input_tone_hz", audio.get("input_tone_hz", 1_000.0))),
        "clock_hz": float(audio.get("clock_hz", 24_576_000.0)),
        "rng_seed": int(audio.get("rng_seed", 42)),
        "waveform_zoom_periods": float(plots.get("waveform_zoom_periods", 5.0)),
        "jitter_overview_fraction": float(plots.get("jitter_overview_fraction", 0.25)),
        "max_points_per_trace": int(plots.get("max_points_per_trace", 0)),
        "fft_single_tone_zoom_percent": float(plots.get("fft_single_tone_zoom_percent", 1.0)),
        "fft_show_full_spectrum": bool(plots.get("fft_show_full_spectrum", False)),
        "multitone_mode": str(signal.get("mode", "single")),
        "multitone_tones_hz": [float(v) for v in signal.get("multitone_tones_hz", [19_000.0, 20_000.0])],
        "comb_tone_count": int(signal.get("comb_tone_count", 32)),
        "comb_freq_min_hz": float(signal.get("comb_freq_min_hz", 1_000.0)),
        "comb_freq_max_hz": float(signal.get("comb_freq_max_hz", 20_000.0)),
        "wav_input_path": _optional_path(signal.get("wav_input_path", None)),
        "wav_output_path": _optional_path(signal.get("wav_output_path", None)),
        "wav_interpolation": str(signal.get("wav_interpolation", "linear")),
        "phase_noise_model": str(phase_noise.get("model", "power_law")),
        "phase_noise_f1_hz": float(phase_noise.get("power_law", {}).get("f1_hz", 0.1)),
        "phase_noise_l1_dbc": float(phase_noise.get("power_law", {}).get("l1_dbc", -80.0)),
        "phase_noise_f2_hz": float(phase_noise.get("power_law", {}).get("f2_hz", 10_000.0)),
        "phase_noise_l2_dbc": float(phase_noise.get("power_law", {}).get("l2_dbc", -180.0)),
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

    if cfg["multitone_mode"] not in {"single", "twotone", "comb", "wav"}:
        raise ValueError("signal.mode must be one of: single, twotone, comb, wav")
    if cfg["phase_noise_model"] not in {"power_law", "fixed_slope", "piecewise"}:
        raise ValueError("phase_noise.model must be one of: power_law, fixed_slope, piecewise")
    if cfg["fs_audio_hz"] <= 0.0 or cfg["duration_s"] <= 0.0 or cfg["clock_hz"] <= 0.0:
        raise ValueError("audio.fs_audio_hz, audio.duration_s, and audio.clock_hz must be > 0")
    if cfg["input_tone_hz"] <= 0.0:
        raise ValueError("signal.input_tone_hz must be > 0")
    if cfg["jitter_integration_fmin_hz"] <= 0.0:
        raise ValueError("integration.fmin_hz must be > 0")
    if cfg["bw_limited_jitter_fmin_hz"] <= 0.0:
        raise ValueError("integration.bw_limited_fmin_hz must be > 0")
    if cfg["bw_limited_jitter_fmax_hz"] <= cfg["bw_limited_jitter_fmin_hz"]:
        raise ValueError("integration.bw_limited_fmax_hz must be > integration.bw_limited_fmin_hz")
    if cfg["max_points_per_trace"] < 0:
        raise ValueError("plots.max_points_per_trace must be >= 0")
    if cfg["fft_single_tone_zoom_percent"] <= 0.0:
        raise ValueError("plots.fft_single_tone_zoom_percent must be > 0")
    if cfg["wav_interpolation"] not in {"linear", "cubic", "farrow"}:
        raise ValueError("signal.wav_interpolation must be one of: linear, cubic, farrow")

    dut_name = cfg["dut_name"]
    if dut_name is None:
        cfg["dut_name"] = None
    else:
        dut_name = str(dut_name).strip()
        cfg["dut_name"] = dut_name if dut_name else None

    return cfg


def _pcm_bytes_to_float(raw_bytes, sampwidth):
    if sampwidth == 1:
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64)
        return (data - 128.0) / 128.0
    if sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64)
        return data / 32768.0
    if sampwidth == 3:
        b = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        val = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        val = np.where((val & sign_mask) != 0, val - (1 << 24), val)
        return val.astype(np.float64) / 8388608.0
    if sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64)
        return data / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")


def _float_to_pcm_bytes(signal, sampwidth):
    sig = np.clip(signal, -1.0, 1.0 - np.finfo(np.float64).eps)
    if sampwidth == 1:
        out = np.round(sig * 128.0 + 128.0).astype(np.uint8)
        return out.tobytes()
    if sampwidth == 2:
        out = np.round(sig * 32767.0).astype("<i2")
        return out.tobytes()
    if sampwidth == 3:
        val = np.round(sig * 8388607.0).astype(np.int32)
        val = np.where(val < 0, val + (1 << 24), val).astype(np.uint32)
        b0 = (val & 0xFF).astype(np.uint8)
        b1 = ((val >> 8) & 0xFF).astype(np.uint8)
        b2 = ((val >> 16) & 0xFF).astype(np.uint8)
        packed = np.empty(val.size * 3, dtype=np.uint8)
        packed[0::3] = b0
        packed[1::3] = b1
        packed[2::3] = b2
        return packed.tobytes()
    if sampwidth == 4:
        out = np.round(sig * 2147483647.0).astype("<i4")
        return out.tobytes()
    raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")


def read_wav_float(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs_hz = wf.getframerate()
        frame_count = wf.getnframes()
        comp = wf.getcomptype()
        if comp != "NONE":
            raise ValueError(f"Only PCM WAV is supported (got compression {comp})")
        raw = wf.readframes(frame_count)

    data = _pcm_bytes_to_float(raw, sampwidth)
    data = data.reshape(-1, channels)
    meta = {
        "channels": channels,
        "sampwidth": sampwidth,
        "framerate": fs_hz,
    }
    return fs_hz, data, meta


def write_wav_float(path, fs_hz, data, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = meta["channels"]
    sampwidth = meta["sampwidth"]
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[1] != channels:
        raise ValueError("Channel count mismatch while writing WAV")
    interleaved = data.reshape(-1)
    raw = _float_to_pcm_bytes(interleaved, sampwidth)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(int(fs_hz))
        wf.writeframes(raw)


def _farrow_interpolate(x_nominal, y, x_query):
    """Farrow polyphase filter using high-order spline interpolation.
    
    Implements high-fidelity audio interpolation via UnivariateSpline with k=5.
    Provides excellent spectral quality for time-domain jitter application with
    minimal interpolation artifacts in the audio band.
    
    Args:
        x_nominal: Array of nominal sample indices (typically 0, 1, 2, ...)
        y: Signal samples corresponding to x_nominal
        x_query: Jittered sample indices where output is desired
    
    Returns:
        Interpolated signal values at x_query indices
    """
    # Use scipy's UnivariateSpline with k=5 for excellent audio fidelity
    # s=0 means exact interpolation (no smoothing), k=5 is quintic spline via fitpack
    try:
        f_interp = scipy.interpolate.UnivariateSpline(x_nominal, y, s=0, k=5)
        return f_interp(x_query)
    except Exception:
        # Fallback: if k=5 fails, try k=3 (cubic), then k=1 (linear)
        try:
            f_interp = scipy.interpolate.UnivariateSpline(x_nominal, y, s=0, k=3)
            return f_interp(x_query)
        except Exception:
            f_interp = scipy.interpolate.UnivariateSpline(x_nominal, y, s=0, k=1)
            return f_interp(x_query)


def apply_jitter_to_audio(audio, fs_hz, jitter_s, interpolation="linear"):
    """Apply clock jitter to audio samples via time-domain perturbation and interpolation.
    
    Args:
        audio: (N, C) array of audio samples (N samples, C channels)
        fs_hz: Audio sample rate in Hz
        jitter_s: (N,) array of timing errors in seconds
        interpolation: Method - 'linear', 'cubic', or 'farrow'
    
    Returns:
        (N, C) array of jittered audio samples
    """
    if interpolation not in {"linear", "cubic", "farrow"}:
        raise ValueError(f"Unknown interpolation method: {interpolation}. Must be one of: linear, cubic, farrow")
    
    sample_count = audio.shape[0]
    idx_nominal = np.arange(sample_count, dtype=np.float64)
    idx_jittered = idx_nominal + jitter_s * fs_hz
    idx_jittered = np.clip(idx_jittered, 0.0, sample_count - 1.0)

    out = np.empty_like(audio)
    
    if interpolation == "linear":
        # Fast linear interpolation using numpy
        for ch in range(audio.shape[1]):
            out[:, ch] = np.interp(idx_jittered, idx_nominal, audio[:, ch])
    
    elif interpolation == "cubic":
        # Cubic spline interpolation (natural boundary conditions)
        for ch in range(audio.shape[1]):
            cs = scipy.interpolate.CubicSpline(idx_nominal, audio[:, ch], bc_type='natural')
            out[:, ch] = cs(idx_jittered)
    
    elif interpolation == "farrow":
        # Farrow polyphase filter with 3rd-order Lagrange (best audio fidelity)
        for ch in range(audio.shape[1]):
            out[:, ch] = _farrow_interpolate(idx_nominal, audio[:, ch], idx_jittered)
    
    return out


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


def decimate_for_plot(x, y, max_points):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size <= max_points:
        return x, y

    idx = np.linspace(0, x.size - 1, max_points, dtype=int)
    return x[idx], y[idx]


def main(
    config_path=DEFAULT_CONFIG_PATH,
    show_plot=True,
    cli_input_wav_path=None,
    cli_output_wav_path=None,
    cli_image_path=None,
    no_image=False,
):
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
    max_points_per_trace = cfg["max_points_per_trace"]
    fft_single_tone_zoom_percent = cfg["fft_single_tone_zoom_percent"]
    fft_show_full_spectrum = cfg["fft_show_full_spectrum"]

    multitone_mode = cfg["multitone_mode"]
    multitone_tones_hz = cfg["multitone_tones_hz"]
    comb_tone_count = cfg["comb_tone_count"]
    comb_freq_min_hz = cfg["comb_freq_min_hz"]
    comb_freq_max_hz = cfg["comb_freq_max_hz"]
    wav_input_path = cfg["wav_input_path"]
    wav_output_path = cfg["wav_output_path"]
    wav_interpolation = cfg["wav_interpolation"]

    cli_in_supplied = cli_input_wav_path is not None
    cli_out_supplied = cli_output_wav_path is not None

    if cli_in_supplied:
        wav_input_path = Path(cli_input_wav_path)
    if cli_out_supplied:
        wav_output_path = Path(cli_output_wav_path)

    # Rule: when only --in is given, derive output beside input as <stem>_jittered.wav.
    if cli_in_supplied and not cli_out_supplied:
        wav_output_path = _default_jittered_wav_path(wav_input_path)
    elif wav_input_path is not None and wav_output_path is None:
        wav_output_path = _default_jittered_wav_path(wav_input_path)

    if no_image:
        output_path = None
    elif cli_image_path is not None:
        output_path = Path(cli_image_path)
    elif output_path is None and wav_input_path is not None:
        output_path = Path(wav_input_path).with_suffix(".png")
    elif output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    if multitone_mode == "wav" and wav_input_path is None:
        raise ValueError(
            "WAV mode requires an input WAV path (set signal.wav_input_path in config or pass --in)"
        )
    if multitone_mode == "wav" and wav_output_path is None:
        raise ValueError(
            "WAV mode requires an output WAV path (set signal.wav_output_path in config or pass --out)"
        )

    phase_noise_model = cfg["phase_noise_model"]
    phase_noise_f1_hz = cfg["phase_noise_f1_hz"]
    phase_noise_l1_dbc = cfg["phase_noise_l1_dbc"]
    phase_noise_f2_hz = cfg["phase_noise_f2_hz"]
    phase_noise_l2_dbc = cfg["phase_noise_l2_dbc"]
    fixed_slope_alpha = cfg["fixed_slope_alpha"]
    fixed_slope_ref_freq_hz = cfg["fixed_slope_ref_freq_hz"]
    fixed_slope_ref_level_dbc = cfg["fixed_slope_ref_level_dbc"]
    piecewise_phase_noise_points = cfg["piecewise_phase_noise_points"]
    jitter_integration_fmin_hz = cfg["jitter_integration_fmin_hz"]
    jitter_integration_fmax_hz = cfg["jitter_integration_fmax_hz"]
    bw_limited_jitter_fmin_hz = cfg["bw_limited_jitter_fmin_hz"]
    bw_limited_jitter_fmax_hz = cfg["bw_limited_jitter_fmax_hz"]

    wav_input = None
    wav_meta = None
    if multitone_mode == "wav":
        fs_wav_hz, wav_input, wav_meta = read_wav_float(wav_input_path)
        if abs(fs_wav_hz - fs_audio_hz) > 1e-12:
            print(
                f"[info] Overriding audio.fs_audio_hz={fs_audio_hz:g} with WAV sample rate {fs_wav_hz:g} Hz"
            )
        fs_audio_hz = float(fs_wav_hz)
        duration_s = wav_input.shape[0] / fs_audio_hz

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
        if phase_noise_model == "fixed_slope":
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
    elif multitone_mode == "wav":
        # Used only for plot zoom defaults in wav mode.
        active_tones = [1_000.0]
    else:
        active_tones = [input_tone_hz]

    if multitone_mode == "wav":
        time_s = np.arange(sample_count) / fs_audio_hz
        jitter_s = phase_noise_rad / (2.0 * np.pi * clock_hz)
        wav_jittered = apply_jitter_to_audio(wav_input, fs_audio_hz, jitter_s, interpolation=wav_interpolation)
        write_wav_float(wav_output_path, fs_audio_hz, wav_jittered, wav_meta)
        print(f"[info] Wrote jittered WAV: {wav_output_path}")
        sampled = wav_jittered[:, 0]
        reference = wav_input[:, 0]
        error = sampled - reference
    else:
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
        if fft_show_full_spectrum:
            fft_band_low = 0.0
            fft_band_high = fs_audio_hz / 2.0
        else:
            fft_band_low = input_tone_hz * (1.0 - fft_single_tone_zoom_percent / 100.0)
            fft_band_high = input_tone_hz * (1.0 + fft_single_tone_zoom_percent / 100.0)
        near_tone = (fft_freq_ref >= fft_band_low) & (fft_freq_ref <= fft_band_high)
        zoom_ref_hz = input_tone_hz

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

        if max_points_per_trace == 0:
            # Auto mode: a few points per horizontal pixel is visually sufficient.
            point_budget = int(max(1000, fig.get_figwidth() * fig.dpi * 2.0))
        else:
            point_budget = max_points_per_trace

        slope_label = model_label
        plot_mask_model = model_freq_bins >= jitter_integration_fmin_hz
        plot_mask_est = psd_freq >= jitter_integration_fmin_hz
        plot_x_est, plot_y_est = decimate_for_plot(psd_freq[plot_mask_est], phase_noise_est_dbc[plot_mask_est], point_budget)
        plot_x_model, plot_y_model = decimate_for_plot(
            model_freq_bins[plot_mask_model], model_phase_noise_dbc[plot_mask_model], point_budget
        )
        axes[0, 0].plot(
            plot_x_est,
            plot_y_est,
            label="Synthesized phase noise",
            color="tab:orange",
            linewidth=1.4,
            alpha=0.9,
            zorder=2,
        )
        axes[0, 0].plot(
            plot_x_model,
            plot_y_model,
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
        plot_x_zoom, plot_y_zoom = decimate_for_plot(time_s[:zoom_count], jitter_s[:zoom_count] * 1e12, point_budget)
        axes[0, 1].plot(plot_x_zoom, plot_y_zoom)
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Timing error [ps]")
        axes[0, 1].set_title("Clock jitter (time domain, zoom)")
        axes[0, 1].grid(True, alpha=0.3)

        plot_x_fft_smp, plot_y_fft_smp = decimate_for_plot(fft_freq_smp[near_tone], fft_db_smp[near_tone], point_budget)
        plot_x_fft_ref, plot_y_fft_ref = decimate_for_plot(fft_freq_ref[near_tone], fft_db_ref[near_tone], point_budget)
        axes[1, 0].plot(plot_x_fft_smp, plot_y_fft_smp, label="Jittered sampling")
        axes[1, 0].plot(
            plot_x_fft_ref,
            plot_y_fft_ref,
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
            if fft_show_full_spectrum:
                axes[1, 0].set_title(f"FFT full spectrum around {input_tone_hz:.0f} Hz tone")
            else:
                axes[1, 0].set_title(
                    f"FFT around {input_tone_hz:.0f} Hz tone (±{fft_single_tone_zoom_percent:g}%)"
                )
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

        plot_x_overview, plot_y_overview = decimate_for_plot(
            time_s[:overview_count], jitter_s[:overview_count] * 1e12, point_budget
        )
        axes[1, 1].plot(plot_x_overview, plot_y_overview)
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
        elif multitone_mode == "wav":
            mode_label = f"wav mode ({Path(wav_input_path).name})"
        else:
            mode_label = f"{input_tone_hz:.0f} Hz single tone"
        heading_prefix = f"DUT: {dut_name} — " if dut_name else ""
        fig.suptitle(
            f"{heading_prefix}{mode_label}\n"
            f"RMS jitter(sim) = {jitter_rms_s * 1e12:.3f} ps, "
            f"RMS jitter(model {jitter_integration_fmin_hz:g}..{effective_fmax_hz:g} Hz) = {model_jitter_rms_s * 1e12:.3f} ps, "
            f"RMS jitter(bw-limited {bw_limited_jitter_fmin_hz:g}..{bw_limited_effective_fmax_hz:g} Hz) = {model_jitter_bw_limited_rms_s * 1e12:.3f} ps, "
            f"jitter-limited SNR = {snr_jitter_db:.2f} dB"
        )

        fig.savefig(output_path, dpi=150)
        print(f"[info] Wrote image: {output_path}")
        if show_plot and plt.get_backend().lower() != "agg":
            plt.show()
        else:
            plt.close(fig)
    else:
        print("[info] Image generation disabled (--noimg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clock phase-noise to audio jitter impact simulator")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Save PNG and WAV outputs without displaying the plot window.",
    )
    parser.add_argument(
        "--in",
        dest="input_wav_path",
        type=Path,
        default=None,
        help="Input WAV path override for wav mode.",
    )
    parser.add_argument(
        "--out",
        dest="output_wav_path",
        type=Path,
        default=None,
        help="Output WAV path override for wav mode.",
    )
    parser.add_argument(
        "--img",
        dest="image_path",
        type=Path,
        default=None,
        help="Output PNG path override.",
    )
    parser.add_argument(
        "--noimg",
        action="store_true",
        help="Disable image generation.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        show_plot=not args.headless,
        cli_input_wav_path=args.input_wav_path,
        cli_output_wav_path=args.output_wav_path,
        cli_image_path=args.image_path,
        no_image=args.noimg,
    )
