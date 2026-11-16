"""Image upscaling core (Real-ESRGAN family, loaded through spandrel).

No web or Hugging Face code lives here, so it runs the same on a GPU server, a hosted Gradio Space
or a plain CPU. Use it from Python or the command line:

    python core.py input.jpg output.png --model general --scale 4

Weights are looked up in this order:
  1. models listed in HF_FILES (the large Nomos DAT model) come from a public Hugging Face model repo;
  2. WEIGHTS_REPO (a Hugging Face model repo holding <id>.safetensors files), if the environment
     variable is set;
  3. the original GitHub release files, downloaded once into WEIGHTS_DIR.
A model whose weights cannot be fetched is skipped by preload() instead of stopping the app.
"""
import argparse
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", Path(__file__).parent / "weights"))
WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", "").strip()

_REL = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_NOMOS = "https://github.com/Phhofm/models/releases/download"

# id -> (label shown to users, file URL or None when the model is hosted on Hugging Face)
MODELS = {
    "general": ("Photo - fast (general-x4v3)", f"{_REL}/v0.2.5.0/realesr-general-x4v3.pth"),
    "photo": ("Photo - detailed (x4plus)", f"{_REL}/v0.1.0/RealESRGAN_x4plus.pth"),
    # Nomos models by Philip Hofmann (CC-BY-4.0, see NOTICE.md), trained on noisy real photos
    "natural": ("Photo - natural texture (Nomos RealPLKSR)", f"{_NOMOS}/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth"),
    "dat": ("Photo - most faithful, heavy (Nomos DAT)", None),
    "anime": ("Anime / illustration (x4plus anime 6B)", f"{_REL}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
}
DEFAULT_MODEL = "general"

# Large models served from a public Hugging Face model repo (a free account can host these).
HF_MODELS_REPO = os.environ.get("HF_MODELS_REPO", "Demassu/fvrom-upscaler-models")
HF_FILES = {"dat": (HF_MODELS_REPO, "dat.safetensors")}

# Per-model tile size in input pixels (default 512). DAT is a transformer and needs far more memory per tile.
TILES = {"dat": 128, "natural": 256}
# DAT is kept out of half precision: transformer attention is the part most likely to misbehave in fp16.
FP32_ONLY = {"dat"}

_loaded = {}


def weights_path(model_id):
    """Return a local path for the model weights, downloading them if needed."""
    if model_id in HF_FILES:
        from huggingface_hub import hf_hub_download

        repo, filename = HF_FILES[model_id]
        return hf_hub_download(repo, filename)
    if WEIGHTS_REPO:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(WEIGHTS_REPO, f"{model_id}.safetensors")
    url = MODELS[model_id][1]
    path = WEIGHTS_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def load_model(model_id):
    """Load a model onto the CPU once and keep it (the GPU is used only while upscaling)."""
    if model_id not in _loaded:
        from spandrel import ModelLoader

        desc = ModelLoader().load_from_file(str(weights_path(model_id)))
        desc.eval()
        _loaded[model_id] = desc
    return _loaded[model_id]


def preload():
    """Load every model; return the ids that loaded. A model whose weights are unavailable is skipped."""
    loaded = []
    for model_id in MODELS:
        try:
            load_model(model_id)
            loaded.append(model_id)
        except Exception as e:  # missing file, offline, repo not uploaded yet
            print(f"[upscaler] skipping {model_id}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    return loaded


@torch.inference_mode()
def _run_tiles(desc, rgb, device, tile, pad, amp=True):
    """Upscale an RGB uint8 array tile by tile, so large images fit in GPU memory."""
    h, w, _ = rgb.shape
    s = desc.scale
    out = np.empty((h * s, w * s, 3), dtype=np.uint8)
    use_amp = amp and device.type == "cuda"
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, h), min(x1 + pad, w)
            t = torch.from_numpy(rgb[ya:yb, xa:xb]).to(device)
            t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                r = desc(t)
            r = r.float().clamp_(0, 1).mul_(255).round_().byte()
            r = r[0].permute(1, 2, 0).cpu().numpy()
            cy, cx = (y0 - ya) * s, (x0 - xa) * s
            out[y0 * s:y1 * s, x0 * s:x1 * s] = r[cy:cy + (y1 - y0) * s, cx:cx + (x1 - x0) * s]
    return out


def upscale(img, model_id=DEFAULT_MODEL, scale=4, tile=None, pad=24):
    """Upscale a PIL image. `scale` is the final factor (2 or 4). `tile` defaults per model (see TILES)."""
    tile = tile or TILES.get(model_id, 512)
    img = ImageOps.exif_transpose(img)
    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA") or "transparency" in img.info else None
    rgb = np.array(img.convert("RGB"))
    desc = load_model(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    desc.to(device)
    try:
        while True:
            try:
                out = _run_tiles(desc, rgb, device, tile, pad, amp=model_id not in FP32_ONLY)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if tile <= 64:
                    raise
                tile //= 2  # retry with smaller tiles
    finally:
        desc.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = Image.fromarray(out)
    size = (img.width * scale, img.height * scale)
    if result.size != size:
        result = result.resize(size, Image.LANCZOS)
    if alpha is not None:
        result.putalpha(alpha.resize(size, Image.LANCZOS))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    p.add_argument("--scale", type=int, default=4, choices=(2, 4))
    p.add_argument("--tile", type=int, default=None, help="tile size in pixels (default depends on the model)")
    a = p.parse_args()
    t0 = time.time()
    res = upscale(Image.open(a.input), a.model, a.scale, a.tile)
    res.save(a.output)
    print(f"{a.input} -> {a.output} {res.size} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
