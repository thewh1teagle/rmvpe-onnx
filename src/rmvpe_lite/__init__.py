__all__ = ["RMVPEOnnx", "RMVPEOnnxConfig", "RMVPETorch"]


def __getattr__(name: str):
    if name in {"RMVPEOnnx", "RMVPEOnnxConfig"}:
        from .onnx import RMVPEOnnx, RMVPEOnnxConfig

        return {"RMVPEOnnx": RMVPEOnnx, "RMVPEOnnxConfig": RMVPEOnnxConfig}[name]
    if name == "RMVPETorch":
        from .torch import RMVPETorch

        return RMVPETorch
    raise AttributeError(f"module 'rmvpe_lite' has no attribute {name!r}")
