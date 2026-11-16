# fvrom-upscaler

A free image upscaler that runs **entirely in your browser**. Drop in a photo and get a 4x (or 2x)
larger version made by a neural network. Your image is never uploaded: the model file is downloaded
once and runs on your own graphics card through WebGPU.

Live site: <https://upscale.fvrom.app>

> **Status.** Working and tested on one machine (see [Measured results](#measured-results)). The
> Detailed and Natural modes were added on 2026-11-16 and may not be on the live site yet. Nothing in
> this repository has been tested on AMD hardware yet.

## What it does

- Upscales JPG, PNG and WebP images 4x or 2x, with transparency kept.
- Three quality modes, each a different model:

| Mode | Model | Good for | Cost |
| --- | --- | --- | --- |
| **Fast** | Real-ESRGAN `realesr-general-x4v3` (1.2M parameters) | Quick results, any WebGPU browser; very small images even without WebGPU | Fastest |
| **Detailed** | Real-ESRGAN `RealESRGAN_x4plus` (16.7M) | Sharper, more natural texture | About 13x slower than Fast on integrated graphics |
| **Natural** | Nomos `4xNomosWebPhoto_RealPLKSR` (7.4M), by Philip Hofmann | Noisy real photos: fur, grain, JPEG damage | **Very slow on integrated graphics** (about 19 s per tile); limited to about 380x380 pixels; not tested on dedicated GPUs |

- Detailed and Natural need a WebGPU GPU with half-precision support; other devices get Fast only.
- Before/after slider and a full-resolution PNG download.
- No account, no server, no upload.

A fourth model, Nomos `4xNomos8kDAT`, is too heavy for a browser and is available only in the
[server version](#server-version-experimental). How the models compare, and what failed, is written up in
[`docs/model-evaluation.md`](docs/model-evaluation.md).

## How it works

1. The page loads [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) and the model file.
2. The image is cut into 192 px tiles with 16 px of mirrored padding, so large images fit in GPU memory.
3. Each tile goes through the model; results are written straight onto the output canvas.
4. Half-precision (fp16) models are used when the GPU supports them. They are about 4x faster in our
   test and differ from full precision by less than 0.2 out of 255 on average. fp16 is a speed and
   download-size choice, not a quality upgrade.
5. Results are cached per mode, so switching modes on the same image is instant.

Size limits: Fast 2.1 megapixels (0.1 on the CPU fallback), Detailed 0.4 megapixels or 20 tiles,
Natural 4 tiles. The output is 16x the input pixel count, so memory limits Fast, and time limits the
other two. Images over the limit get a warning and a "Shrink to fit" button.

## Measured results

One machine only: an Intel integrated GPU (gen-12lp), a Chromium-based browser, Windows 11. These are
single runs, not a controlled benchmark. Details and method in [`docs/benchmarks.md`](docs/benchmarks.md).

| Test | Result |
| --- | --- |
| ONNX vs PyTorch, full precision | max difference 4.5e-6 (Fast), 2.2e-6 (Detailed) |
| 576x576 image, Fast, fp32 vs fp16 | 9.2 s vs 2.1 s |
| 576x576 image, Detailed fp16 | 28.4 s |
| 384x384 image, Natural fp16 (4 tiles) | 74.9 s warm, about 18.7 s per tile |
| 1673x1254 image, Fast fp16 (to 6692x5016) | 14.9 s |
| CPU fallback, one 128x128 tile | 14.7 s |

On a noisy real photograph, both Nomos models looked clearly better than Detailed; on a harsh synthetic
test there was no clear winner. See [`docs/model-evaluation.md`](docs/model-evaluation.md).

## Run it locally

WebGPU needs a secure context, and `localhost` counts as one:

```bash
python -m http.server 8000 --directory web
# open http://localhost:8000 in Chrome or Edge
```

Any static host works for deployment (Netlify, GitHub Pages, Cloudflare Pages). The `web/` folder is
the whole site. The page loads ONNX Runtime Web from the jsDelivr CDN.

## Rebuild the model files

`tools/export_onnx.py` converts a model to ONNX, checks it against PyTorch, and writes a half-precision
copy:

```bash
pip install torch spandrel onnx onnxruntime onnxscript onnxconverter-common
python tools/export_onnx.py                            # Fast     -> web/model/general*.onnx
python tools/export_onnx.py --model photo --fp16-only   # Detailed -> web/model/photo_fp16.onnx
python tools/export_onnx.py --model natural --fp16-only # Natural  -> web/model/realplksr_fp16.onnx
```

The original weights are downloaded from the authors' releases into your temp folder.

## Server version (experimental)

`server/` holds all the models, including DAT, behind a small [Gradio](https://www.gradio.app) app, for
platforms that host Gradio apps or for running on a GPU server:

```bash
pip install -r server/requirements.txt
python server/app.py
```

- DAT's weights are served from a public Hugging Face model repo (default `Demassu/fvrom-upscaler-models`,
  override with `HF_MODELS_REPO`). Until that repo is populated, the app skips DAT and shows the others.
  `tools/publish_to_hf.py` converts DAT to safetensors, checks the copy is identical, writes a credited
  model card, and uploads it (you supply your own `HF_TOKEN`).
- It has only been tested on the CPU. It has **not** been run on CUDA or on AMD ROCm yet.

## Known limits

- Tested only in a Chromium-based browser on one Intel integrated GPU. Safari, Firefox, phones and
  discrete GPUs are untested.
- Detailed is slow on integrated graphics (about 3 s per 192 px tile there), and Natural far slower
  (about 19 s). Why Natural is so slow on WebGPU is not understood yet.
- Neural upscalers invent texture. Do not use upscaled images as evidence of fine detail.
- Browsers without WebGPU get Fast mode on the CPU, which is very slow.

## Roadmap

Not done yet; nothing below has results:

- Find out why Natural (RealPLKSR) is so slow on WebGPU, and test it on a dedicated GPU.
- Run the server version on AMD ROCm and on CUDA, and publish real numbers.
- Publish the DAT weights to Hugging Face (the tool is ready).

## Credits and licenses

The code is MIT licensed ([`LICENSE`](LICENSE)). The models are Real-ESRGAN by Xintao Wang et al.
(BSD-3-Clause) and Nomos RealPLKSR / DAT by Philip Hofmann (CC-BY-4.0 as listed on OpenModelDB), all
converted to other file formats. See [`NOTICE.md`](NOTICE.md) for the full notices, including a note
about the Nomos license wording.
