# fvrom-upscaler

A free image upscaler that runs **entirely in your browser**. Drop in a photo and get a 4x (or 2x)
larger version made by a neural network. Your image is never uploaded: the model file is downloaded
once and runs on your own graphics card through WebGPU.

Live site: <https://upscale.fvrom.app>

> **Status.** Working and tested on one machine (see [Measured results](#measured-results)). Detailed
> mode was added on 2026-11-16 and may not be on the live site yet. Nothing in this repository has
> been tested on AMD hardware yet.

## What it does

- Upscales JPG, PNG and WebP images 4x or 2x, with transparency kept.
- **Fast mode**: Real-ESRGAN `realesr-general-x4v3` (1.2M parameters). Works on any browser with
  WebGPU, and falls back to the CPU for very small images.
- **Detailed mode**: Real-ESRGAN `RealESRGAN_x4plus` (16.7M parameters). More natural texture, about
  13x slower in our test. Only offered on WebGPU GPUs that support half precision.
- Before/after slider and a full-resolution PNG download.
- No account, no server, no upload.

## How it works

1. The page loads [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) and the model file.
2. The image is cut into 192 px tiles with 16 px of mirrored padding, so large images fit in GPU memory.
3. Each tile goes through the model; results are written straight onto the output canvas.
4. Half-precision (fp16) models are used when the GPU supports them. They are about 4x faster in our
   test and differ from full precision by less than 0.2 out of 255 on average. fp16 is a speed and
   download-size choice, not a quality upgrade.
5. Results are cached per mode, so switching Fast and Detailed on the same image is instant.

Size limits (input pixels): Fast 2.1 MP, Detailed 0.4 MP, CPU fallback 0.1 MP. The output is 16x the
input pixel count, so memory is the limit, and Detailed is limited by time.

## Measured results

One machine only: an Intel integrated GPU (gen-12lp), a Chromium-based browser, Windows 11. These are
single runs, not a controlled benchmark. Details and method in [`docs/benchmarks.md`](docs/benchmarks.md).

| Test | Result |
| --- | --- |
| ONNX vs PyTorch, full precision | max difference 4.5e-6 (Fast), 2.2e-6 (Detailed) |
| 576x576 image, Fast, fp32 vs fp16 | 9.2 s vs 2.1 s |
| 576x576 image, Detailed fp16 | 28.4 s |
| 1673x1254 image, Fast fp16 (to 6692x5016) | 14.9 s |
| CPU fallback, one 128x128 tile | 14.7 s |

## Run it locally

WebGPU needs a secure context, and `localhost` counts as one:

```bash
python -m http.server 8000 --directory web
# open http://localhost:8000 in Chrome or Edge
```

Any static host works for deployment (Netlify, GitHub Pages, Cloudflare Pages). The `web/` folder is
the whole site. The page loads ONNX Runtime Web from the jsDelivr CDN.

## Rebuild the model files

`tools/export_onnx.py` converts a Real-ESRGAN model to ONNX, checks it against PyTorch, and writes a
half-precision copy:

```bash
pip install torch spandrel onnx onnxruntime onnxscript onnxconverter-common
python tools/export_onnx.py                           # Fast model -> web/model/general*.onnx
python tools/export_onnx.py --model photo --fp16-only # Detailed model -> web/model/photo_fp16.onnx
```

The original weights are downloaded from the Real-ESRGAN releases into your temp folder.

## Server version (experimental)

`server/` holds the same models behind a small [Gradio](https://www.gradio.app) app, for platforms
that host Gradio apps or for running on a GPU server:

```bash
pip install -r server/requirements.txt
python server/app.py
```

It has only been tested on the CPU. It has **not** been run on CUDA or on AMD ROCm yet.

## Known limits

- Tested only in a Chromium-based browser on one Intel integrated GPU. Safari, Firefox, phones and
  discrete GPUs are untested.
- Detailed mode is slow on integrated graphics (about 3 s per 192 px tile there).
- Neural upscalers invent texture. On one test photo the Detailed model made a convincing but
  invented pattern on a microphone mesh. Do not use upscaled images as evidence of fine detail.
- Browsers without WebGPU get Fast mode on the CPU, which is very slow.

## Roadmap

Not done yet; nothing below has results:

- Evaluate the Nomos family of photo models (`4xNomosWebPhoto_RealPLKSR`, `4xNomos8kDAT`) against
  `x4plus` on realistic photos, then add the ones that work in the browser as extra options.
- Host larger model files in a public Hugging Face model repository instead of in this repo.
- Run the server version on AMD ROCm and publish real numbers.

## Credits and licenses

The code is MIT licensed ([`LICENSE`](LICENSE)). The models are Real-ESRGAN by Xintao Wang et al.
(BSD-3-Clause), converted to ONNX. See [`NOTICE.md`](NOTICE.md) for the full notices.
