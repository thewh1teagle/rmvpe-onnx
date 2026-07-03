from __future__ import annotations

from pathlib import Path

import numpy as np

from .model.constants import SAMPLE_RATE
from .model.inference import RMVPE as _TorchRMVPE


class RMVPETorch:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        hop_length: int = 160,
    ) -> None:
        self.device = device
        self.model = _TorchRMVPE(str(model_path), device=device, hop_length=hop_length)

    def extract(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        threshold: float = 0.03,
        use_viterbi: bool = False,
    ) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError("audio must be a mono 1D array")
        return self.model.infer_from_audio(
            audio.astype(np.float32, copy=False),
            sample_rate=sample_rate,
            device=self.device,
            thred=threshold,
            use_viterbi=use_viterbi,
        )


__all__ = ["RMVPETorch", "SAMPLE_RATE"]
