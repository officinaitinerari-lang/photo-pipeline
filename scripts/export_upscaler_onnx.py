"""Build-time helper: convert Real-ESRGAN `realesr-general-x4v3.pth` (BSD-3) to ONNX.

Runs only in the Docker builder stage (needs torch); the runtime image ships just the .onnx.
The SRVGGNetCompact architecture below mirrors basicsr/realesrgan (num_feat=64, num_conv=32, x4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64, num_conv: int = 32, upscale: int = 4) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    model = SRVGGNetCompact()
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.rand(1, 3, 64, 64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(args.output), opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h4", 3: "w4"}},
    )
    with torch.no_grad():
        y = model(dummy)
    assert tuple(y.shape) == (1, 3, 256, 256), y.shape
    print(f"exported {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
