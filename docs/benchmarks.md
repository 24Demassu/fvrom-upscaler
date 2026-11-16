# Benchmarks and checks

Measured on 2026-11-16. Everything here comes from one machine, so treat it as indicative.

**Setup:** Intel integrated GPU (WebGPU adapter vendor `intel`, architecture `gen-12lp`, `shader-f16`
supported), Chromium-based browser (version 156), Windows 11, ONNX Runtime Web 1.22.0, tiles of
192 px plus 16 px padding. "Cold" runs include loading the model and compiling shaders; "warm" runs
do not. Each figure is a single run.

Test images were ordinary photographs that are not included in this repository; only their size
affects the timings.

## Accuracy

| Check | Result |
| --- | --- |
| Fast model, ONNX fp32 vs PyTorch (random 90x130 input) | max difference 4.4e-6 to 4.5e-6 |
| Detailed model, ONNX fp32 vs PyTorch (random 90x130 input) | max difference 2.2e-6 |
| Fast model, fp16 page output vs PyTorch, 368x490 photo | mean difference 0.18 of 255 |
| Detailed model, fp16 alone vs PyTorch, 240x240 crop, no tiling | mean difference 0.031 of 255 |
| Detailed model, page pipeline (fp16 + tiling) vs whole-image PyTorch | mean difference 1.36 of 255. This comes from tiling (each tile sees only 16 px of context), not from fp16: the same tiling in fp32 PyTorch gives 1.36 |
| Visible seams at tile borders | none seen in a side-by-side of the tile boundary (difference map amplified 10x showed only faint scattered noise) |

fp16 is a speed and size choice. It did not measurably change quality for these two models. For any
future model, especially transformer-based ones, fp16 must be re-checked against fp32.

## Speed (Intel integrated GPU, WebGPU)

| Image | Mode | Tiles | Time |
| --- | --- | --- | --- |
| 576x576 | Fast, fp32 | 9 | 9.2 s |
| 576x576 | Fast, fp16 | 9 | 2.1 s |
| 240x240 crop | Fast, fp16 | 4 | 1.6 s |
| 240x240 crop | Detailed, fp16, cold | 4 | 11.1 s |
| 576x576 | Detailed, fp16, cold | 9 | 27.3 s |
| 576x576 | Detailed, fp16, warm | 9 | 28.4 s |
| 771x777 (0.6 MP) to 3084x3108 | Detailed, fp16 | 25 | 71.9 s |
| 1673x1254 (2.1 MP) to 6692x5016 | Fast, fp16 | 63 | 14.9 s |
| 128x128 | Fast, fp32, CPU (WASM, single thread) | 1 | 14.7 s |

Detailed costs about 2.9 s per tile and Fast about 0.2 to 0.4 s per tile here, so Detailed is roughly
13x slower. Tile count rounds up, so time depends more on tile count than on megapixels. The Detailed
input limit is set to 0.4 MP (about 16 tiles) for that reason.

## Not measured

- Other browsers, phones, discrete GPUs, and non-Intel GPUs (including AMD).
- The server version on any GPU.
- Multi-threaded CPU mode (needs cross-origin isolation headers).
- fp16 vs fp32 on models other than the two above.
