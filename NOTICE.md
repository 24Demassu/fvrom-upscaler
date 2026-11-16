# Third-party notices

The code in this repository is MIT licensed (see `LICENSE`). The **model files** and the libraries
below are not covered by that license; they keep their own.

## Model files in `web/model/`

| File | What it is | Source | License |
| --- | --- | --- | --- |
| `general.onnx`, `general_fp16.onnx` | Real-ESRGAN `realesr-general-x4v3`, converted to ONNX (and to half precision for the `_fp16` file) | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), release v0.2.5.0 | BSD-3-Clause (full text below) |
| `photo_fp16.onnx` | Real-ESRGAN `RealESRGAN_x4plus`, converted to ONNX and to half precision | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), release v0.1.0 | BSD-3-Clause (full text below) |
| `realplksr_fp16.onnx` | **4xNomosWebPhoto_RealPLKSR** by Philip Hofmann, converted to ONNX and to half precision | [Phhofm/models release](https://github.com/Phhofm/models/releases/tag/4xNomosWebPhoto_RealPLKSR) (also [OpenModelDB](https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR)) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see the note below) |

These files are modified copies: only the file format and number precision were changed, using
`tools/export_onnx.py`. The network weights are the original authors' work.

### Nomos models: attribution and a license note

`4xNomosWebPhoto_RealPLKSR` was created by **Philip Hofmann** (GitHub: Phhofm; Helaman on OpenModelDB),
trained on the Nomos-v2 dataset with neosr. OpenModelDB lists the license as **CC-BY-4.0**. The author's
own release page reads "CC-BY-0.4", which is not an existing license version and appears to be a typo,
and the author's repository states no license. We treat the model as CC-BY-4.0: we credit the author,
link the source and license, and state that the file was converted (to ONNX and half precision). If the
author prefers different terms or would like the file removed, please open an issue and it will be
changed or removed.

The same applies to `4xNomos8kDAT` (also by Philip Hofmann, CC-BY-4.0 on OpenModelDB, a fine-tune of the
official DAT model by Zheng Chen et al., code under Apache-2.0, ICCV 2023). A converted copy for the
server version is hosted on Hugging Face, prepared by `tools/publish_to_hf.py`; the model card there
carries the same credits. The authors of the official DAT pretrained weights state no separate terms.

## Libraries

| Library | Use | License |
| --- | --- | --- |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | Runs the models in the browser. Loaded from the jsDelivr CDN at run time, not copied into this repository | MIT |
| [spandrel](https://github.com/chaiNNer-org/spandrel) | Loads PyTorch model files in `tools/` and `server/` | MIT |
| PyTorch, Gradio, ONNX, ONNX Runtime | Used by `tools/` and `server/` when you run them; not redistributed here | See each project |

## Real-ESRGAN license (BSD-3-Clause)

BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
