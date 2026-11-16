"""Prepare the large Nomos DAT model for a public Hugging Face model repo, and optionally upload it.

    python tools/publish_to_hf.py --pth 4xNomos8kDAT.pth --out hf-upload            # convert + check + model card
    HF_TOKEN=hf_... python tools/publish_to_hf.py --pth 4xNomos8kDAT.pth --out hf-upload --upload

What it does:
  1. loads the original .pth through spandrel (which uses a restricted unpickler) and saves the weights,
     unchanged, as dat.safetensors (a format that cannot run code);
  2. loads the .safetensors again and checks it produces the same output as the original;
  3. writes a model card (README.md) that credits the authors and states the license and the change;
  4. with --upload, creates the repo (public) and uploads the folder. The token is read from the
     HF_TOKEN environment variable and is never printed or stored.

A free Hugging Face account can host public model repos; only Spaces need PRO.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from spandrel import ModelLoader

DEFAULT_REPO = "Demassu/fvrom-upscaler-models"

MODEL_CARD = """---
license: cc-by-4.0
library_name: pytorch
tags:
  - super-resolution
  - image-upscaling
  - real-esrgan
  - dat
---

# Converted copy of 4xNomos8kDAT

This repository holds `dat.safetensors`, a **format conversion only** of **4xNomos8kDAT** (a 4x photo
upscaler). The weights are unchanged; the original `.pth` file was re-saved as `.safetensors`, which
cannot execute code when loaded. It is used by [fvrom-upscaler](https://github.com/24Demassu/fvrom-upscaler).

## Credits

- **Model: 4xNomos8kDAT by Philip Hofmann** (Helaman on OpenModelDB). Trained on the Nomos8k dataset as a
  fine-tune of the official DAT model. Original: <https://openmodeldb.info/models/4x-Nomos8kDAT>
  (author's collection: <https://github.com/Phhofm/models>).
- **Architecture: DAT, Dual Aggregation Transformer** by Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong,
  Xiaokang Yang and Fisher Yu (ICCV 2023), Apache-2.0 code: <https://github.com/zhengchen1999/DAT>

```bibtex
@inproceedings{chen2023dual,
  title={Dual Aggregation Transformer for Image Super-Resolution},
  author={Chen, Zheng and Zhang, Yulun and Gu, Jinjin and Kong, Linghe and Yang, Xiaokang and Yu, Fisher},
  booktitle={ICCV},
  year={2023}
}
```

## License

CC-BY-4.0, as listed for this model on OpenModelDB. Attribution is given above, and the change made
(format conversion) is stated above. The authors of the official DAT pretrained weights state no separate
terms for them. If you are the author and would like this copy changed or removed, please open a discussion.

## Use

```python
from huggingface_hub import hf_hub_download
from spandrel import ModelLoader
model = ModelLoader().load_from_file(hf_hub_download("%(repo)s", "dat.safetensors"))
```

DAT is a heavy transformer: use small tiles (128 px) and expect it to be slow on a CPU.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pth", required=True, help="path to the original 4xNomos8kDAT.pth")
    p.add_argument("--out", required=True, help="folder to write dat.safetensors and README.md into")
    p.add_argument("--repo", default=DEFAULT_REPO, help="target Hugging Face repo id")
    p.add_argument("--upload", action="store_true", help="create the repo and upload (needs HF_TOKEN)")
    a = p.parse_args()
    torch.set_num_threads(3)  # keep the machine usable

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    desc = ModelLoader().load_from_file(a.pth); desc.eval()
    print(f"loaded: {desc.architecture}, scale x{desc.scale}, {sum(p.numel() for p in desc.model.parameters()) / 1e6:.2f}M parameters")
    sd = {k: v.detach().contiguous().cpu().float() for k, v in desc.model.state_dict().items()}
    target = out / "dat.safetensors"
    save_file(sd, str(target))
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB, {len(sd)} tensors)")

    # round trip: the converted file must give the same output as the original
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        ref = desc(x)
        again = ModelLoader().load_from_file(str(target)); again.eval()
        got = again(x)
    diff = (ref - got).abs().max().item()
    print(f"round-trip check: max abs difference {diff:.2e}")
    if diff > 1e-5:
        sys.exit("converted file does not match the original; not publishing")

    (out / "README.md").write_text(MODEL_CARD % {"repo": a.repo}, encoding="utf-8")
    print(f"wrote {out / 'README.md'}")

    if a.upload:
        token = (os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'")
        if not token:
            sys.exit("--upload needs the HF_TOKEN environment variable (a token with write access).")
        # Catch placeholder text before it reaches the network. The value is never printed.
        if not token.startswith("hf_") or len(token) < 20 or any(c.isspace() for c in token):
            sys.exit("HF_TOKEN does not look like a Hugging Face token (a real one starts with 'hf_', is one word "
                     "with no spaces, and is created at https://huggingface.co/settings/tokens with write access). "
                     "Did you leave placeholder text in it? Nothing was uploaded.")
        from huggingface_hub import HfApi

        try:
            api = HfApi(token=token)
            api.create_repo(a.repo, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="model",
                              commit_message="Add dat.safetensors (converted copy of 4xNomos8kDAT)")
        except Exception as e:  # print only the error type: messages can echo request headers
            sys.exit(f"Upload failed ({type(e).__name__}). Check that the token is valid, has write access, "
                     "and that you are online. Nothing about the token was printed.")
        print(f"uploaded: https://huggingface.co/{a.repo}")
    else:
        print("not uploaded (add --upload with HF_TOKEN set to publish)")


if __name__ == "__main__":
    main()
