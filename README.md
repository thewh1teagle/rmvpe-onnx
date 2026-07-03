## RMVPE Lite

Lightweight RMVPE F0/pitch extraction for TTS and voice model training, with
ONNXRuntime and PyTorch backends.

Install the ONNX backend from GitHub:

```bash
uv pip install "rmvpe-lite[onnx] @ git+https://github.com/thewh1teagle/rmvpe-onnx"
```

Or install the PyTorch backend:

```bash
uv pip install "rmvpe-lite[torch] @ git+https://github.com/thewh1teagle/rmvpe-onnx"
```

Download the ONNX model from the [model-files-v1.0 release](https://github.com/thewh1teagle/rmvpe-onnx/releases/download/model-files-v1.0/rmvpe.onnx).

ONNX usage:

```python
import soundfile as sf
from rmvpe_lite.onnx import RMVPEOnnx

audio, sample_rate = sf.read("audio.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

model = RMVPEOnnx("rmvpe.onnx")
f0 = model.extract(audio, sample_rate=sample_rate)
```

`f0` is a NumPy array of pitch values in Hz. Unvoiced frames are `0.0`.

See `examples/basic_onnx.py` and `examples/basic_torch.py` for complete examples.

See [Attribution](docs/attribution.md) for RMVPE code, model, and paper sources.
