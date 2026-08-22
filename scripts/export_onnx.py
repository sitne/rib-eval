#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_transformer import WinPredictor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MODEL = OUT / "transformer_pooled.pt"
ONNX_PATH = OUT / "transformer_pooled.onnx"


def main():
    model = WinPredictor()
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    model.eval()

    x = torch.randn(2, 44, 10, 18, dtype=torch.float32)
    at = torch.zeros(2, dtype=torch.int64)

    torch.onnx.export(
        model,
        (x, at),
        str(ONNX_PATH),
        input_names=["x", "at"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "at": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported -> {ONNX_PATH}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping parity check (uv add onnx onnxruntime)")
        return

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ref = model(x, at).detach().numpy()
    out = sess.run(["logits"], {"x": x.numpy(), "at": at.numpy()})[0]
    diff = float(np.max(np.abs(ref - out)))
    print(f"parity: max |torch - onnx| = {diff:.2e}")
    assert diff < 1e-4, f"parity check failed: {diff}"


if __name__ == "__main__":
    main()
