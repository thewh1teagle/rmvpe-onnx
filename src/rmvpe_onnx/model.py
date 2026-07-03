from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class RMVPEConfig:
    sample_rate: int
    n_mels: int
    window_length: int
    hop_length: int
    mel_fmin: float
    mel_fmax: float
    n_class: int
    cents_const: float
    center: bool
    stft_pad_mode: str


class RMVPE:
    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.config = _config_from_metadata(
            self.session.get_modelmeta().custom_metadata_map
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.window_length,
            n_mels=self.config.n_mels,
            fmin=self.config.mel_fmin,
            fmax=self.config.mel_fmax,
            htk=True,
        ).astype(np.float32)

    def extract(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        threshold: float = 0.03,
        use_viterbi: bool = False,
    ) -> np.ndarray:
        audio = _mono_float32(audio)
        if sample_rate != self.config.sample_rate:
            audio = _resample_audio(audio, sample_rate, self.config.sample_rate)

        mel = self._log_mel(audio)
        frame_count = mel.shape[-1]
        mel = _pad_frames_to_multiple(mel, multiple=32)
        hidden = self.session.run(
            [self.output_name],
            {self.input_name: mel[None, :, :].astype(np.float32, copy=False)},
        )[0]
        hidden = hidden[:, :frame_count, :]

        if use_viterbi:
            return _to_viterbi_f0(hidden, self.config, threshold)
        return _to_local_average_f0(hidden, self.config, threshold)

    def _log_mel(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        if audio.size == 0:
            raise ValueError("audio must contain at least one sample")

        magnitude = np.abs(
            librosa.stft(
                audio,
                n_fft=cfg.window_length,
                hop_length=cfg.hop_length,
                win_length=cfg.window_length,
                window="hann",
                center=cfg.center,
                pad_mode=cfg.stft_pad_mode,
            )
        )
        mel = self._mel_basis @ magnitude
        return np.log(np.clip(mel, 1e-5, None)).astype(np.float32, copy=False)


def _config_from_metadata(metadata: dict[str, str]) -> RMVPEConfig:
    required = {
        "sample_rate",
        "n_mels",
        "window_length",
        "hop_length",
        "mel_fmin",
        "mel_fmax",
        "n_class",
        "cents_const",
        "center",
        "stft_pad_mode",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"ONNX model is missing RMVPE metadata: {joined}")

    return RMVPEConfig(
        sample_rate=int(metadata["sample_rate"]),
        n_mels=int(metadata["n_mels"]),
        window_length=int(metadata["window_length"]),
        hop_length=int(metadata["hop_length"]),
        mel_fmin=float(metadata["mel_fmin"]),
        mel_fmax=float(metadata["mel_fmax"]),
        n_class=int(metadata["n_class"]),
        cents_const=float(metadata["cents_const"]),
        center=metadata["center"].lower() == "true",
        stft_pad_mode=metadata["stft_pad_mode"],
    )


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono 1D array")
    return audio.astype(np.float32, copy=False)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample_rate values must be positive")
    if audio.size == 0 or source_rate == target_rate:
        return audio
    return librosa.resample(
        audio, orig_sr=source_rate, target_sr=target_rate, res_type="scipy"
    ).astype(np.float32, copy=False)


def _pad_frames_to_multiple(mel: np.ndarray, *, multiple: int) -> np.ndarray:
    frame_count = mel.shape[-1]
    padded_count = multiple * ((frame_count - 1) // multiple + 1)
    pad_count = padded_count - frame_count
    if pad_count == 0:
        return mel
    mode = "reflect" if frame_count > 1 else "edge"
    return np.pad(mel, ((0, 0), (0, pad_count)), mode=mode)


def _to_local_average_f0(
    hidden: np.ndarray,
    config: RMVPEConfig,
    threshold: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    hidden = hidden[0]
    bins = np.arange(config.n_class, dtype=np.float32)
    cents_mapping = bins * 20.0 + config.cents_const

    if center is None:
        center = np.argmax(hidden, axis=1)
    f0 = np.zeros(hidden.shape[0], dtype=np.float32)

    for frame_index, center_bin in enumerate(center):
        start = max(0, int(center_bin) - 4)
        end = min(config.n_class, int(center_bin) + 5)
        weights = hidden[frame_index, start:end]
        if weights.size == 0 or float(np.max(hidden[frame_index])) < threshold:
            continue
        weight_sum = float(np.sum(weights))
        if weight_sum == 0:
            continue
        cents = float(np.sum(weights * cents_mapping[start:end]) / weight_sum)
        f0[frame_index] = 10.0 * (2.0 ** (cents / 1200.0))

    return f0


def _to_viterbi_f0(
    hidden: np.ndarray,
    config: RMVPEConfig,
    threshold: float,
) -> np.ndarray:
    salience = hidden[0].astype(np.float64, copy=False)
    probability = salience.T / np.maximum(salience.T.sum(axis=0, keepdims=True), 1e-12)
    center = librosa.sequence.viterbi(
        probability, _viterbi_transition(config.n_class)
    ).astype(np.int64)
    return _to_local_average_f0(hidden, config, threshold, center=center)


def _viterbi_transition(n_class: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(n_class), np.arange(n_class))
    transition = np.maximum(30 - np.abs(x - y), 0).astype(np.float64)
    return transition / transition.sum(axis=1, keepdims=True)
