# Model evaluation

Four 4x upscalers were compared on 2026-11-16, on one laptop. This page records what was tested, what
was seen, what failed, and the decisions that followed. No test images are included in this repository:
the public-domain sample photos are not republished here, and the other photograph is not ours to
publish.

| Model | Architecture | Parameters | Where it is used |
| --- | --- | --- | --- |
| Real-ESRGAN `realesr-general-x4v3` | Compact conv net | 1.21M | Browser, **Fast** |
| Real-ESRGAN `RealESRGAN_x4plus` | RRDBNet | 16.70M | Browser, **Detailed** |
| Nomos `4xNomosWebPhoto_RealPLKSR` | RealPLKSR | 7.39M | Browser, **Natural** (slow on integrated graphics); server |
| Nomos `4xNomos8kDAT` | DAT (transformer) | 14.80M | Server only; weights on Hugging Face |

The Nomos models are by Philip Hofmann and are credited in [`NOTICE.md`](../NOTICE.md).

## Test A: synthetic damage on public-domain photos

Four sample photographs (astronaut, cat, coffee cup, rocket). Each was cropped to a multiple of 4, shrunk
4x (bicubic), and JPEG-compressed at quality 75 to imitate a web photo. Each model upscaled it 4x and
was compared with the original. CPU, PyTorch, default threads.

| | PSNR (dB) | SSIM | Time per photo |
| --- | --- | --- | --- |
| bicubic resize | 25.77 | 0.7227 | 0 s |
| Fast (general-x4v3) | 25.00 | 0.7317 | 0.2 s |
| Detailed (x4plus) | 24.85 | 0.7168 | 3.6 s |
| Nomos RealPLKSR | 24.79 | 0.6823 | 2.2 s |
| Nomos DAT | 25.84 | 0.7308 | 10.2 s |

**PSNR and SSIM cannot rank these models.** Plain bicubic resizing beat both Real-ESRGAN models, because
these models are trained to look sharp and realistic, and PSNR rewards blur. The scores are recorded
only for completeness.

By eye, there was no clear winner. Detailed was the crispest but invented texture and changed the eyes of
a face that was only about 40 pixels wide. RealPLKSR gave the best fur on the cat but made odd eyes and
blotchy patches. DAT was the most faithful and least inventive but visibly soft. This test is harsh:
the 4x shrink plus JPEG removes detail that no model can restore.

## Test B: a real, noisy photograph at native resolution

One photograph of a rodeo scene with fur, dust, gravel and sensor noise. Two 192x192 crops at native
resolution, no degradation: horse mane and fur, and gravel and dust. Judged by eye; there is no ground
truth. CPU, PyTorch, 3 threads.

| | Mane and fur crop | Gravel crop |
| --- | --- | --- |
| Fast | 0.9 s | 0.9 s |
| Detailed (x4plus) | 13.5 s | 14.1 s |
| Nomos RealPLKSR | 9.5 s | 10.0 s |
| Nomos DAT | 33.1 s | 35.2 s |

What was seen:

- **Fast** smeared the hair into soft blur and flattened the gravel.
- **Detailed** was sharper but smudgy: noise was turned into blotches.
- **Nomos RealPLKSR** rebuilt fine individual hair strands, kept shadows clean, and gave natural grain
  with sharp small stones.
- **Nomos DAT** also rebuilt fine strands and grain, with slightly "etched" lines in dark areas.

On this kind of input both Nomos models were clearly better than Detailed. That matches their listing:
they were trained on realistic noise, lens blur and JPEG/WebP damage. It is two crops of one photo, so
treat it as a strong hint, not a measurement. All of these models invent plausible texture; do not use
an upscaled image as evidence of real fine detail.

## Test C: can they run in the browser?

**RealPLKSR: yes, but slowly.**

- Exports to ONNX with dynamic sizes: 29.9 MB in fp32, 15.3 MB in fp16. Output matches PyTorch to 3e-6,
  8e-6 and 6e-5 at 64x64, 90x130 and 224x224 inputs.
- fp16 against PyTorch on a 224x224 tile: mean difference 1.7e-4, worst 2.9e-3 (on a 0 to 1 scale), no NaN.
- All 23 operator types it uses are in the ONNX Runtime Web WebGPU kernel list.
- **Speed on WebGPU** (Intel integrated GPU, fp16, 384x384 image = 4 tiles): 76.8 s on the first run and
  74.9 s on the second, about **18.7 s per tile**. Detailed takes about 2.9 s per tile on the same GPU,
  so RealPLKSR is about 6.5x slower in the browser even though it is about 30% faster on the CPU.
  The cause was not diagnosed. Large-kernel convolutions and the many shape operations in the exported
  graph (2,316 nodes) are suspects.
- Not tested on a dedicated GPU, where it may be much faster.

**DAT: not practical in the browser.**

- A dynamic-size export produced a 72.0 MB graph with 19,544 nodes, but it only runs at the size it was
  traced at (64x64). At 90x130 and 224x224 it fails in a Reshape node.
- Converting that graph to fp16 fails (a type error in a Cast node).
- Exporting at a fixed 224x224 was killed by the operating system for lack of memory on a 15.7 GB
  laptop.
- On the CPU DAT is the slowest of the four: 33 to 35 s for a 192x192 crop.

## Decisions

- **Browser:** three modes. **Fast** and **Detailed** as before, plus **Natural** (RealPLKSR). Natural is
  labelled as very slow on integrated graphics and is limited to 4 tiles (about 380x380 pixels) per
  image. All three models are fp16 files shipped in this repository (`web/model/`).
- **Server (`server/`):** all models, including DAT. DAT's weights are hosted in a public Hugging Face
  model repo (a free account can host these; only Spaces need PRO), prepared with
  `tools/publish_to_hf.py`. DAT runs in full precision with 128 px tiles.
- **Open question:** why RealPLKSR is so slow on WebGPU, and whether it is fast on dedicated GPUs.
- **Not done:** running any of this on AMD ROCm or CUDA. No results exist yet.

## Licenses

See [`NOTICE.md`](../NOTICE.md). The Nomos models are listed as CC-BY-4.0 on OpenModelDB. The author's
own release page writes "CC-BY-0.4", which is not a real version and appears to be a typo; the author's
repository states no license. We treat it as CC-BY-4.0, credit the author, link the source and state
that the files were converted. The official DAT pretrained weights state no separate terms.
