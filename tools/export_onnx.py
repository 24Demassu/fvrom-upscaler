"""Export a Real-ESRGAN model to ONNX for the in-browser upscaler, and check it against PyTorch.

    python export_onnx.py                 # "general" (realesr-general-x4v3) -> web/model/general.onnx
                                          # and web/model/general_fp16.onnx (half precision, ~4x faster on WebGPU)
    python export_onnx.py --model photo --fp16-only   # x4plus ("Detailed" on the page): only
                                          # web/model/photo_fp16.onnx (34 MB) is kept; the 67 MB fp32 file is discarded

Needs: torch, spandrel, onnx, onnxruntime, onnxscript, onnxconverter-common.
The ONNX file has dynamic height and width, input and output are float32 NCHW RGB in 0..1
(the fp16 file keeps float32 inputs and outputs too). Weights come from the official Real-ESRGAN
release (downloaded once into the system temp folder). The web/ folder is what gets published,
so this script lives outside it.
"""
import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "server"))
import core  # noqa: E402


def local_weights(model_id):
    url = core.MODELS[model_id][1]
    path = Path(tempfile.gettempdir()) / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="general", choices=list(core.MODELS))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16-only", action="store_true",
                   help="publish only the fp16 file (the fp32 one is written to a temp folder and discarded); "
                        "used for big models like x4plus whose fp32 file is too large to ship")
    a = p.parse_args()

    from spandrel import ModelLoader

    desc = ModelLoader().load_from_file(str(local_weights(a.model)))
    desc.eval()
    net = desc.model.eval()
    scale = desc.scale
    print(f"Loaded {a.model}: scale x{scale}, {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")

    out_dir = HERE.parent / "web" / "model"
    out_dir.mkdir(exist_ok=True)
    fp32_dir = Path(tempfile.mkdtemp()) if a.fp16_only else out_dir
    out_path = fp32_dir / f"{a.model}.onnx"

    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "oh", 3: "ow"}},
        opset_version=a.opset, do_constant_folding=True, dynamo=False,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Check ONNX Runtime output against PyTorch on a size the export did not see.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = torch.rand(1, 3, 90, 130)
    with torch.inference_mode():
        ref = net(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = np.abs(ref - got).max()
    print(f"Output shape {got.shape}; max abs difference vs PyTorch: {diff:.2e}")
    if diff > 1e-3:
        sys.exit("ONNX output does not match PyTorch")
    print("OK")

    # Half-precision copy for GPUs that support shader-f16 (the page picks it automatically).
    import onnx
    from onnxconverter_common import float16

    fp16_path = out_dir / f"{a.model}_fp16.onnx"
    onnx.save(float16.convert_float_to_float16(onnx.load(str(out_path)), keep_io_types=True), str(fp16_path))
    got16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    diff16 = np.abs(ref - got16).max()
    print(f"Wrote {fp16_path} ({fp16_path.stat().st_size / 1e6:.2f} MB); max abs difference vs PyTorch: {diff16:.2e}")
    if diff16 > 1e-2:
        sys.exit("fp16 ONNX output differs too much from PyTorch")
    if a.fp16_only:
        import shutil

        shutil.rmtree(fp32_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
