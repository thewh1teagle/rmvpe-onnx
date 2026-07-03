"""
wget https://github.com/thewh1teagle/rmvpe-onnx/releases/download/model-files-v1.0/rmvpe.onnx
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav
uv run --extra onnx examples/basic_onnx.py
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from rmvpe_lite.onnx import RMVPEOnnx


def main() -> None:
    model_path = Path("rmvpe.onnx")
    audio_path = Path("female1.wav")

    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    model = RMVPEOnnx(model_path)
    f0 = model.extract(audio, sample_rate=sample_rate)
    voiced = f0[f0 > 0]

    print(f"frames: {len(f0)}")
    print(f"voiced: {len(voiced)}")
    if voiced.size:
        print(f"f0 min: {voiced.min():.2f} Hz")
        print(f"f0 mean: {voiced.mean():.2f} Hz")
        print(f"f0 max: {voiced.max():.2f} Hz")


if __name__ == "__main__":
    main()
