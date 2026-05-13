# Audio Clock Jitter Simulator

This project simulates how clock phase noise maps into sampling jitter and degrades audio signal quality.

Main script: `clock_audio_jitter.py`
Default configuration: `clock_audio_jitter_config.yaml`
Output image path: configured by `plots.output_path`

## What It Does

- Builds a phase-noise profile from one of three models:
  - `power_law` (two anchors)
  - `fixed_slope` (specified slope and reference point)
  - `piecewise` (frequency/level anchor list with interpolation)
- Synthesizes phase noise in the frequency domain
- Converts phase noise to timing jitter
- Applies jitter to sample timing for single-tone, 2-tone, or 32-tone comb input
- Computes and reports:
  - simulation RMS jitter
  - model RMS jitter over analysis band (`integration.fmin_hz..integration.fmax_hz`, Nyquist-clipped)
  - model RMS jitter over bw-limited band (`integration.bw_limited_fmin_hz..integration.bw_limited_fmax_hz`, Nyquist-clipped)
  - jitter-limited SNR
- Plots:
  - phase-noise profile (ideal model vs synthesized)
  - shaded bw-limited integration band on the phase-noise plot
  - short jitter zoom
  - FFT around the signal or full-band (multitone/comb)
  - zoomed-out jitter trend for low-frequency behavior

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`
- `pyyaml`

Install:

```bash
pip install numpy matplotlib pyyaml
```

## Run

Default config file in project root:

```bash
python clock_audio_jitter.py
```

Use a custom config:

```bash
python clock_audio_jitter.py --config my_config.yaml
```

Tip: all reported integration upper bounds are effectively clipped to audio Nyquist (`audio.fs_audio_hz / 2`).

## Sample Output

Generic crystal example output:

![Generic crystal piecewise single-tone example](images/generic-crystal_piecewise_single_sample.png)

This sample is generated from [configs/generic_crystal.yaml](configs/generic_crystal.yaml). It uses the `piecewise` phase-noise model with typical crystal-like close-in and far-out anchors, then reports jitter over a bw-limited band (`integration.bw_limited_fmin_hz` to `integration.bw_limited_fmax_hz`) to mimic a basic oscillator specification near 1 ps RMS.

## Preset Configurations

Ready-to-run examples are in [configs](configs):

- [configs/generic_crystal.yaml](configs/generic_crystal.yaml): piecewise single-tone profile intended to emulate a generic ~1 ps RMS bw-limited crystal oscillator
- [configs/power_law_single.yaml](configs/power_law_single.yaml): single 1 kHz tone with 2-anchor power-law phase noise
- [configs/piecewise_twotone_19k_20k.yaml](configs/piecewise_twotone_19k_20k.yaml): 19+20 kHz two-tone with piecewise phase-noise profile
- [configs/piecewise_comb_32tone.yaml](configs/piecewise_comb_32tone.yaml): 32-tone comb with piecewise phase-noise profile

Run them directly:

```bash
python clock_audio_jitter.py --config configs/generic_crystal.yaml
python clock_audio_jitter.py --config configs/power_law_single.yaml
python clock_audio_jitter.py --config configs/piecewise_twotone_19k_20k.yaml
python clock_audio_jitter.py --config configs/piecewise_comb_32tone.yaml
```

## Configuration Guide

All runtime settings are in `clock_audio_jitter_config.yaml`. Config keys are grouped by function:

- `audio`: sampling clock/audio timing and repeatability controls
- `signal`: what test stimulus is generated (`single`, `twotone`, or `comb`)
- `phase_noise`: how $L(f)$ is modeled (power-law, fixed-slope, or piecewise)
- `integration`: frequency bands used for reported RMS jitter metrics
- `plots`: figure layout, decimation, and output path

### Simple Example

```yaml
dut_name: "Example XO"

audio:
  fs_audio_hz: 48000.0
  duration_s: 20.0
  clock_hz: 24576000.0
  rng_seed: 42

signal:
  mode: single
  input_tone_hz: 1000.0

phase_noise:
  model: piecewise
  piecewise_points:
    - [0.1, -80.0]
    - [10.0, -110.0]
    - [1000.0, -145.0]
    - [10000.0, -155.0]

integration:
  fmin_hz: 0.1
  fmax_hz: 24000.0
  bw_limited_fmin_hz: 100.0
  bw_limited_fmax_hz: 24000.0

plots:
  waveform_zoom_periods: 5.0
  jitter_overview_fraction: 0.25
  max_points_per_trace: 0
  fft_single_tone_zoom_percent: 1.0
  fft_show_full_spectrum: false
  output_path: output/example.png
```

How this example behaves:

- Generates a 1 kHz single-tone test signal.
- Shapes phase noise from the piecewise $L(f)$ anchors (values are dBc/Hz).
- Reports full-band jitter over `0.1..24000 Hz` and bw-limited jitter over `100..24000 Hz`.
- Saves the final 2x2 summary figure to `output/example.png`.

### Top-Level

- `dut_name` (optional): label shown in the figure heading as `DUT: <name>`. If omitted or empty, no DUT label is shown.

### `audio`

- `fs_audio_hz`: audio sample rate in Hz.
- `duration_s`: simulation length in seconds.
- `clock_hz`: sampling clock frequency used for phase-to-time conversion.
- `rng_seed`: random seed for repeatability.

### `signal`

- `mode`: `single`, `twotone`, or `comb`.
- `input_tone_hz`: used when `mode: single`.
- `multitone_tones_hz`: list of tones used when `mode: twotone`.
- `comb_tone_count`, `comb_freq_min_hz`, `comb_freq_max_hz`: comb generator controls used when `mode: comb`.

### `phase_noise`

- `model`: `power_law`, `fixed_slope`, or `piecewise`.

`power_law` fields:

- `f1_hz`, `l1_dbc`, `f2_hz`, `l2_dbc`.
- Defines a straight line in log-frequency vs dB, converted to $S_\phi(f)$ internally.

`fixed_slope` fields:

- `alpha`: slope exponent in $S_\phi(f)=K/f^\alpha$.
- `ref_freq_hz`, `ref_level_dbc`: reference anchor point used to solve for $K$.

`piecewise_points`:

- List of `[offset_frequency_hz, level_dbc_per_hz]`.
- Frequencies must be > 0 and strictly increasing.
- At least two points are required.
- Interpolation is linear in log-frequency and dBc/Hz.
- Values outside the anchor range are endpoint-clamped.

### `integration`

- `fmin_hz`, `fmax_hz`: primary analysis integration band for model RMS jitter.
- `bw_limited_fmin_hz`, `bw_limited_fmax_hz`: secondary bw-limited integration band.

Both integration bands are limited by Nyquist (`audio.fs_audio_hz / 2`) in the current implementation.

### `plots`

- `waveform_zoom_periods`: short-time jitter panel zoom.
- `jitter_overview_fraction`: long-time jitter panel coverage as fraction of run.
- `max_points_per_trace`: plot decimation budget per trace (`0` = auto based on figure resolution).
- `fft_single_tone_zoom_percent`: single-tone FFT zoom width as +/- percent around `signal.input_tone_hz`.
- `fft_show_full_spectrum`: when `true`, single-tone FFT uses full `0..Nyquist` instead of zooming.
- `output_path`: output image path.

## Notes

- For low-frequency-dominated phase noise, increase `audio.duration_s` and `plots.jitter_overview_fraction`.
- The phase-noise panel shades the bw-limited jitter band with a discrete tint.
- The idealized model trace is drawn in the foreground for readability.
- The summary includes simulation jitter plus both analysis-band and bw-limited model jitter values.
- In `twotone` mode, the script marks the IMD2 difference product in the FFT panel.
- In `comb` mode, FFT panel shows full audio Nyquist band.
