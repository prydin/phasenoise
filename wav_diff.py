"""Compute the RMS difference between two WAV files and report it in dBc."""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from clock_audio_jitter import read_wav_float
except ImportError:
    import wave

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
            val = np.where((val & (1 << 23)) != 0, val - (1 << 24), val)
            return val.astype(np.float64) / 8388608.0
        if sampwidth == 4:
            data = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64)
            return data / 2147483648.0
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    def read_wav_float(path):
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            fs_hz = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        data = _pcm_bytes_to_float(raw, sampwidth)
        return fs_hz, data.reshape(-1, channels), {"channels": channels, "sampwidth": sampwidth, "framerate": fs_hz}


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMS difference between two WAV files and report in dBc."
    )
    parser.add_argument("reference", type=Path, help="Reference (clean) WAV file")
    parser.add_argument("modified", type=Path, help="Modified (jittered) WAV file")
    args = parser.parse_args()

    fs_ref, audio_ref, meta_ref = read_wav_float(args.reference)
    fs_mod, audio_mod, meta_mod = read_wav_float(args.modified)

    if abs(fs_ref - fs_mod) > 1:
        print(f"ERROR: Sample rate mismatch ({fs_ref} Hz vs {fs_mod} Hz)", file=sys.stderr)
        sys.exit(1)

    if meta_ref["channels"] != meta_mod["channels"]:
        print(
            f"ERROR: Channel count mismatch ({meta_ref['channels']} vs {meta_mod['channels']})",
            file=sys.stderr,
        )
        sys.exit(1)

    n = min(audio_ref.shape[0], audio_mod.shape[0])
    if audio_ref.shape[0] != audio_mod.shape[0]:
        print(
            f"WARNING: Length mismatch ({audio_ref.shape[0]} vs {audio_mod.shape[0]} samples), "
            f"comparing first {n} samples."
        )

    ref = audio_ref[:n]
    mod = audio_mod[:n]
    diff = mod - ref

    ref_rms = np.sqrt(np.mean(ref ** 2))
    diff_rms = np.sqrt(np.mean(diff ** 2))

    if ref_rms < 1e-20:
        print("ERROR: Reference signal has zero RMS — cannot compute dBc.", file=sys.stderr)
        sys.exit(1)

    diff_dbc = 20.0 * np.log10(diff_rms / ref_rms)

    print(f"Reference : {args.reference.name}")
    print(f"Modified  : {args.modified.name}")
    print(f"Samples   : {n}  ({n / fs_ref:.3f} s at {fs_ref} Hz)")
    print(f"Channels  : {meta_ref['channels']}")
    print()
    print(f"Reference RMS : {ref_rms:.6e}")
    print(f"Error RMS     : {diff_rms:.6e}")
    print(f"Error         : {diff_dbc:+.2f} dBc")

    if meta_ref["channels"] > 1:
        print()
        print("Per-channel breakdown:")
        for ch in range(meta_ref["channels"]):
            ch_ref_rms = np.sqrt(np.mean(ref[:, ch] ** 2))
            ch_diff_rms = np.sqrt(np.mean(diff[:, ch] ** 2))
            ch_dbc = 20.0 * np.log10(ch_diff_rms / ch_ref_rms) if ch_ref_rms > 1e-20 else float("nan")
            print(f"  Channel {ch}: ref={ch_ref_rms:.4e}  err={ch_diff_rms:.4e}  {ch_dbc:+.2f} dBc")


if __name__ == "__main__":
    main()
